#!/usr/bin/env python3
"""
evaluation/calibrate.py
Data-driven threshold sweep & calibration for Mini AI Knowledge Assistant.
Measures Hit@1, Hit@3, Recall@5, MRR, In-Scope Coverage, Out-of-Scope Abstention,
and F1 across candidate thresholds.
Outputs evaluation/results/retrieval_calibration.json and
docs/figures/threshold_calibration.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from embeddings.sentence_transformer import BGESentenceTransformerEmbedder
from evaluation.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.relevance import evaluate_relevance
from retrieval.retriever import DenseRetriever
from retrieval.vector_store import VectorStore


def load_dataset(dataset_path: Path) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_calibration() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    index_dir = repo_root / settings.INDEX_DIR
    dataset_path = repo_root / "evaluation" / "dataset.yaml"
    output_dir = repo_root / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = repo_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RETRIEVAL EVALUATION & THRESHOLD CALIBRATION")
    print(f"Index Directory: {index_dir}")
    print(f"Dataset File:    {dataset_path}")
    print("=" * 80)

    # 1. Load Vector Store & Embedder
    store = VectorStore.load_index(index_dir)
    embedder = BGESentenceTransformerEmbedder()
    retriever = DenseRetriever(store=store, embedder=embedder)

    # 2. Load Evaluation Dataset
    dataset = load_dataset(dataset_path)
    questions = dataset.get("questions", [])

    in_scope_items = [q for q in questions if q.get("in_scope", True)]
    out_scope_items = [q for q in questions if not q.get("in_scope", True)]

    print(f"Loaded {len(questions)} evaluation questions:")
    print(f"  - In-Scope Questions:     {len(in_scope_items)}")
    print(f"  - Out-of-Scope Questions: {len(out_scope_items)}")

    # 3. Pre-run retrieval (K=5) for all questions
    in_scope_results = []
    for item in in_scope_items:
        retrieved = retriever.retrieve(item["question"], top_k=5)
        in_scope_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected_evidence"],
                "retrieved": retrieved,
                "top_score": retrieved[0].score if retrieved else 0.0,
            }
        )

    out_scope_results = []
    for item in out_scope_items:
        retrieved = retriever.retrieve(item["question"], top_k=5)
        out_scope_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieved": retrieved,
                "top_score": retrieved[0].score if retrieved else 0.0,
            }
        )

    # 4. Calculate Base Metrics (without threshold filtering)
    h1_scores = [
        hit_at_k(res["retrieved"], res["expected"], k=1) for res in in_scope_results
    ]
    h3_scores = [
        hit_at_k(res["retrieved"], res["expected"], k=3) for res in in_scope_results
    ]
    r5_scores = [
        recall_at_k(res["retrieved"], res["expected"], k=5) for res in in_scope_results
    ]
    rr_scores = [
        reciprocal_rank(res["retrieved"], res["expected"]) for res in in_scope_results
    ]

    base_hit1 = sum(h1_scores) / len(h1_scores) if h1_scores else 0.0
    base_hit3 = sum(h3_scores) / len(h3_scores) if h3_scores else 0.0
    base_recall5 = sum(r5_scores) / len(r5_scores) if r5_scores else 0.0
    base_mrr = mean_reciprocal_rank(rr_scores)

    print("\n[BASE DENSE RETRIEVAL QUALITY (TOP-K=5)]")
    print(f"  - Hit@1:    {base_hit1 * 100:.1f}%")
    print(f"  - Hit@3:    {base_hit3 * 100:.1f}%")
    print(f"  - Recall@5: {base_recall5 * 100:.1f}%")
    print(f"  - MRR:      {base_mrr:.3f}")

    # 5. Threshold Sweep (0.05 to 0.70)
    thresholds = np.linspace(0.05, 0.70, 66)
    sweep_data = []

    best_f1 = -1.0
    best_thresh = settings.RELEVANCE_THRESHOLD

    for t in thresholds:
        t_val = round(float(t), 3)

        # In-scope evaluation
        in_proceed_count = 0
        in_correct_evidence = 0
        for res in in_scope_results:
            decision = evaluate_relevance(res["retrieved"], t_val)
            if decision.decision == "proceed":
                in_proceed_count += 1
                if hit_at_k(decision.selected_chunks, res["expected"], k=1) > 0:
                    in_correct_evidence += 1

        in_coverage = (
            in_proceed_count / len(in_scope_results) if in_scope_results else 0.0
        )
        in_precision = (
            in_correct_evidence / in_proceed_count if in_proceed_count > 0 else 0.0
        )

        # Out-of-scope evaluation
        out_abstain_count = 0
        for res in out_scope_results:
            decision = evaluate_relevance(res["retrieved"], t_val)
            if decision.decision == "abstain":
                out_abstain_count += 1

        out_abstain_rate = (
            out_abstain_count / len(out_scope_results) if out_scope_results else 0.0
        )

        # Overall F1 score of the gate
        # Recall of gate = in_coverage, Precision of gate = out_abstain_rate
        precision_gate = out_abstain_rate
        recall_gate = in_coverage
        if precision_gate + recall_gate > 0:
            f1_gate = (
                2 * (precision_gate * recall_gate) / (precision_gate + recall_gate)
            )
        else:
            f1_gate = 0.0

        if f1_gate > best_f1:
            best_f1 = f1_gate
            best_thresh = t_val

        sweep_data.append(
            {
                "threshold": t_val,
                "in_scope_coverage": round(in_coverage, 4),
                "in_scope_precision": round(in_precision, 4),
                "out_scope_abstention_rate": round(out_abstain_rate, 4),
                "f1_gate_score": round(f1_gate, 4),
            }
        )

    print("\n[THRESHOLD CALIBRATION RESULT]")
    print(f"  - Configured Default Threshold: {settings.RELEVANCE_THRESHOLD:.2f}")
    print(
        f"  - Calibrated Optimal Threshold: {best_thresh:.2f}"
        f" (F1 Gate Score: {best_f1:.4f})"
    )

    # 6. Generate Plot
    plt.figure(figsize=(10, 6))
    t_vals = [s["threshold"] for s in sweep_data]
    cov_vals = [s["in_scope_coverage"] * 100 for s in sweep_data]
    abs_vals = [s["out_scope_abstention_rate"] * 100 for s in sweep_data]
    f1_vals = [s["f1_gate_score"] * 100 for s in sweep_data]

    plt.plot(
        t_vals, cov_vals, label="In-Scope Coverage (%)", color="#1f77b4", linewidth=2
    )
    plt.plot(
        t_vals,
        abs_vals,
        label="Out-of-Scope Abstention Rate (%)",
        color="#2ca02c",
        linewidth=2,
    )
    plt.plot(
        t_vals,
        f1_vals,
        label="Gate F1 Score (%)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )

    plt.axvline(
        x=best_thresh,
        color="red",
        linestyle=":",
        label=f"Calibrated Threshold ({best_thresh:.2f})",
    )
    plt.title("Retrieval Relevance Threshold Calibration Curve", fontsize=14, pad=12)
    plt.xlabel("Similarity Threshold (Cosine)", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    plot_path = figures_dir / "threshold_calibration.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  - Saved Calibration Plot:       {plot_path}")

    # 7. Write Calibration Results JSON
    results_payload = {
        "metadata": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "vector_dim": settings.VECTOR_DIM,
            "top_k": settings.TOP_K,
            "in_scope_count": len(in_scope_items),
            "out_scope_count": len(out_scope_items),
            "configured_threshold": settings.RELEVANCE_THRESHOLD,
            "calibrated_threshold": best_thresh,
        },
        "base_metrics": {
            "hit_at_1": round(base_hit1, 4),
            "hit_at_3": round(base_hit3, 4),
            "recall_at_5": round(base_recall5, 4),
            "mrr": round(base_mrr, 4),
        },
        "threshold_sweep": sweep_data,
    }

    json_path = output_dir / "retrieval_calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"  - Saved Calibration Data:       {json_path}")
    print("=" * 80)

    return results_payload


if __name__ == "__main__":
    run_calibration()
