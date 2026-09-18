#!/usr/bin/env python3
"""
scripts/build_index.py
Build, persist, reload, and inspect the local FAISS vector store and document registry.
Reuses Phase 01 document ingestion pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from embeddings.sentence_transformer import BGESentenceTransformerEmbedder
from ingestion.chunker import bge_token_counter, chunk_cleaned_pages
from ingestion.cleaner import clean_pages
from ingestion.loader import load_pdf_pages
from kb.store import DocumentRegistry
from retrieval.vector_store import VectorStore


def print_status(index_dir: Path) -> None:
    faiss_path = index_dir / "index.faiss"
    manifest_path = index_dir / "manifest.json"

    print("=" * 80)
    print("VECTOR INDEX STATUS REPORT")
    print(f"Index Directory: {index_dir.resolve()}")
    print("=" * 80)

    try:
        store = VectorStore.load_index(index_dir)
        registry = DocumentRegistry(index_dir)
        faiss_size_bytes = faiss_path.stat().st_size if faiss_path.exists() else 0

        print("State:                 READY")
        print(f"Documents Ingested:    {len(registry.documents)}")
        print(f"Total Chunks:          {len(store.chunks)}")
        print(f"FAISS Vectors:         {store.num_vectors}")
        print(f"Vector Dimension:      {store.dimension}")
        print(f"Embedding Model:       {store.embedding_model}")
        print(f"Index Type:            {store.index_type}")
        print(f"FAISS File Size:       {faiss_size_bytes / 1024.0:.1f} KB")
        print(f"Manifest File:         {manifest_path.resolve()}")
        print(f"Index File:            {faiss_path.resolve()}")
        print("-" * 80)
        print("Ingested Documents:")
        for doc in registry.list_documents():
            details = f"({doc.pages} pages, {doc.chunks} chunks)"
            print(f"  * [{doc.doc_id}] {doc.filename} {details}")
    except Exception as exc:
        print(f"State:                 EMPTY / NOT BUILT ({exc})")
        print(f"Index File Exists:     {faiss_path.exists()}")
        print(f"Manifest File Exists:  {manifest_path.exists()}")
    print("=" * 80)


def build_index(force: bool = False) -> None:
    docs_dir = Path(settings.DOCUMENTS_DIR)
    index_dir = Path(settings.INDEX_DIR)

    pdf_files = sorted(list(docs_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF documents found in '{docs_dir.resolve()}'.")
        return

    print("=" * 80)
    print("BUILDING FAISS VECTOR STORE")
    print(f"Source Documents Directory: {docs_dir.resolve()}")
    print(f"Target Index Directory:     {index_dir.resolve()}")
    print(f"Found {len(pdf_files)} PDF file(s).")
    print("=" * 80)

    registry = DocumentRegistry(index_dir)

    # Load existing store if not force rebuild
    store: VectorStore
    if not force and (index_dir / "manifest.json").exists():
        try:
            store = VectorStore.load_index(index_dir)
            print("Loaded existing compatible vector store.")
        except Exception as exc:
            print(f"Notice: Existing index not usable ({exc}). Building fresh store.")
            store = VectorStore()
    else:
        store = VectorStore()

    embedder = BGESentenceTransformerEmbedder()
    new_chunks_count = 0
    new_docs_count = 0

    for pdf_path in pdf_files:
        file_bytes = pdf_path.read_bytes()
        filename = pdf_path.name

        # Extraction -> Cleaning -> Chunking
        pages = load_pdf_pages(file_bytes, filename)
        cleaned_pages, _ = clean_pages(pages)
        doc_chunks = chunk_cleaned_pages(
            pages=cleaned_pages,
            filename=filename,
            file_bytes=file_bytes,
            token_counter=bge_token_counter,
        )

        if not doc_chunks:
            print(f"Skipping '{filename}': no chunks generated.")
            continue

        doc_id = doc_chunks[0].doc_id

        # Deduplication check
        if registry.is_duplicate(doc_id) and not force:
            print(f"Skipping '{filename}' [doc_id={doc_id}] (already indexed).")
            continue

        print(f"Indexing '{filename}' [doc_id={doc_id}]: {len(doc_chunks)} chunks...")
        texts = [c.text for c in doc_chunks]
        vectors = embedder.embed_chunks(texts)

        # Add chunks and vectors to store
        store.add_chunks(doc_chunks, vectors)

        # Register document
        registry.register(
            doc_id=doc_id,
            filename=filename,
            pages=len(pages),
            chunks=len(doc_chunks),
        )
        new_chunks_count += len(doc_chunks)
        new_docs_count += 1

    # Persist atomically
    store.save_index(index_dir)
    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print(f"Newly Ingested Docs:   {new_docs_count}")
    print(f"Newly Ingested Chunks: {new_chunks_count}")
    print(f"Total FAISS Vectors:   {store.num_vectors}")
    print(f"Index Directory:       {index_dir.resolve()}")
    print("=" * 80)


def main():
    desc = "Build and inspect FAISS vector store."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--status", action="store_true", help="Display index status report and exit"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild of index from scratch"
    )
    args = parser.parse_args()

    index_dir = Path(settings.INDEX_DIR)

    if args.status:
        print_status(index_dir)
    else:
        build_index(force=args.force)


if __name__ == "__main__":
    main()
