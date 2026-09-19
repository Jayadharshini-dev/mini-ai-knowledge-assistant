#!/usr/bin/env python3
"""
scripts/retrieve.py
CLI tool to execute dense vector retrieval for a question and display Top-K chunks,
cosine similarity scores, metadata, and relevance/abstention decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from embeddings.sentence_transformer import BGESentenceTransformerEmbedder
from retrieval.relevance import evaluate_relevance
from retrieval.retriever import DenseRetriever
from retrieval.vector_store import VectorStore


def inspect_retrieval(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> None:
    index_dir = Path(settings.INDEX_DIR)
    k_val = top_k or settings.TOP_K
    thresh_val = threshold if threshold is not None else settings.RELEVANCE_THRESHOLD

    print("=" * 80)
    print("DENSE VECTOR RETRIEVAL INSPECTOR")
    print(f"Index Directory: {index_dir.resolve()}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"Configured K:   {k_val}")
    print(f"Threshold:      {thresh_val:.2f}")
    print("=" * 80)
    print(f'\nQUESTION: "{question}"')

    # Load store & embedder
    store = VectorStore.load_index(index_dir)
    embedder = BGESentenceTransformerEmbedder()
    retriever = DenseRetriever(store=store, embedder=embedder)

    # Retrieve candidates
    chunks = retriever.retrieve(question=question, top_k=k_val, threshold=thresh_val)

    # Evaluate relevance & abstention decision
    relevance_decision = evaluate_relevance(chunks, threshold=thresh_val)

    print(f"\nRELEVANCE DECISION: [{relevance_decision.decision.upper()}]")
    print(f"Top Similarity Score: {relevance_decision.top_score:.4f}")
    print(f"Message: {relevance_decision.message}")

    print(f"\nTOP-{len(chunks)} CANDIDATE CHUNKS:")
    print("-" * 80)

    for chunk in chunks:
        status_flag = (
            "PASS [>= Threshold]" if chunk.above_threshold else "FAIL [< Threshold]"
        )
        print(
            f"Rank #{chunk.rank} | Score: {chunk.score:.4f} | Status: {status_flag}"
        )
        print(
            f"Doc: {chunk.chunk.document} | Page: {chunk.chunk.page} |"
            f" ID: {chunk.chunk.chunk_id}"
        )
        preview = chunk.chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f'  Excerpt: "{preview}"')
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect dense vector retrieval for a question."
    )
    parser.add_argument("question", type=str, help="Question string to retrieve for")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-K candidates to retrieve"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Relevance threshold override"
    )
    args = parser.parse_args()

    inspect_retrieval(args.question, args.top_k, args.threshold)


if __name__ == "__main__":
    main()
