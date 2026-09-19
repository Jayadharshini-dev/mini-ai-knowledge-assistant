from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.config import settings

EXPECTED_CALIBRATION_SHA256 = (
    "c1b1d00a8826bb2c70d637eb9fdc4845104a106292a3b084daa3d3c8c50b155d"
)


def test_phase03_calibration_artifact_integrity_and_sha256():
    repo_root = Path(__file__).resolve().parent.parent
    calib_path = repo_root / "evaluation" / "results" / "retrieval_calibration.json"

    assert calib_path.exists(), f"Missing calibration artifact at '{calib_path}'"

    raw_bytes = calib_path.read_bytes()
    computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    assert computed_sha256 == EXPECTED_CALIBRATION_SHA256, (
        f"SHA256 mismatch! Expected {EXPECTED_CALIBRATION_SHA256}, "
        f"got {computed_sha256}"
    )

    data = json.loads(raw_bytes.decode("utf-8"))
    meta = data.get("metadata", {})
    base = data.get("base_metrics", {})

    assert meta.get("configured_threshold") == 0.32
    assert meta.get("calibrated_threshold") == 0.67
    assert meta.get("top_k") == 4
    assert meta.get("in_scope_count") == 20
    assert meta.get("out_scope_count") == 10

    assert base.get("hit_at_1") == 0.8
    assert base.get("hit_at_3") == 1.0
    assert base.get("recall_at_5") == 1.0
    assert base.get("mrr") == 0.8833


def test_phase09_experiment_artifact_relationship_validation():
    repo_root = Path(__file__).resolve().parent.parent
    exp_path = repo_root / "evaluation" / "results" / "retrieval_experiments.json"

    assert exp_path.exists(), f"Missing experiments artifact at '{exp_path}'"

    with open(exp_path, encoding="utf-8") as f:
        data = json.load(f)

    exps = data.get("experiments", {})
    assert len(exps) == 3, f"Expected 3 experiments, found {len(exps)}"
    assert "production_baseline" in exps
    assert "unnormalized_embeddings_ablation" in exps
    assert "pre_calibration_threshold_comparison" in exps

    # Production baseline checks
    baseline = exps["production_baseline"]
    assert baseline["configuration"]["threshold"] == 0.67
    assert baseline["configuration"]["normalized_embeddings"] is True
    assert baseline["metrics"]["hit_at_1"] == 0.8
    assert baseline["metrics"]["hit_at_3"] == 1.0
    assert baseline["metrics"]["recall_at_5"] == 1.0
    assert baseline["metrics"]["mrr"] == 0.8833
    assert baseline["metrics"]["gating_applicable"] is True

    # Unnormalized ablation checks
    unnorm = exps["unnormalized_embeddings_ablation"]
    assert unnorm["configuration"]["normalized_embeddings"] is False
    assert unnorm["metrics"]["gating_applicable"] is False
    assert unnorm["metrics"]["gating_metrics"] is None
    assert "observed_raw_score_range" in unnorm["metrics"]

    # Pre-calibration threshold comparison checks
    thresh032 = exps["pre_calibration_threshold_comparison"]
    assert thresh032["configuration"]["threshold"] == 0.32
    assert thresh032["metrics"]["comparison_type"] == "gating_calibration"
    assert thresh032["metrics"]["shares_retrieval_with"] == "production_baseline"
    assert thresh032["metrics"]["gating_applicable"] is True

    # Preservation block checks
    preservation = data.get("phase03_preservation", {})
    assert preservation.get("calibration_sha256") == EXPECTED_CALIBRATION_SHA256
    assert preservation.get("configured_threshold") == 0.32
    assert preservation.get("calibrated_threshold") == 0.67
    assert preservation.get("metrics_match") is True
    assert preservation.get("calibration_file_rewritten") is False


def test_phase09_evidence_artifact_completeness_and_coherence():
    """Verify evidence artifact completeness across 30 benchmark queries.

    Note: Evaluation evidence artifact logs K=5 candidates per query (ranks 1..5)
    for evaluation metrics (e.g. Recall@5), while production runtime uses TOP_K=4.
    """
    repo_root = Path(__file__).resolve().parent.parent
    evidence_path = repo_root / "evaluation" / "results" / "retrieval_evidence.json"

    assert evidence_path.exists(), f"Missing evidence artifact at '{evidence_path}'"

    with open(evidence_path, encoding="utf-8") as f:
        questions = json.load(f)

    assert len(questions) == 30, f"Expected 30 questions, got {len(questions)}"

    question_ids = [q["question_id"] for q in questions]
    assert len(set(question_ids)) == 30, "Question IDs are not unique!"

    in_scope_count = sum(1 for q in questions if q["in_scope"])
    out_scope_count = sum(1 for q in questions if not q["in_scope"])

    assert in_scope_count == 20
    assert out_scope_count == 10

    required_candidate_fields = {
        "rank",
        "chunk_id",
        "document",
        "page",
        "chunk_index",
        "score",
        "above_threshold",
        "is_relevant",
        "text_preview",
        "text_truncated",
    }

    for q in questions:
        cands = q.get("retrieved_candidates", [])
        assert len(cands) == 5, (
            f"Question '{q['question_id']}' has {len(cands)} candidates, expected 5"
        )

        ranks = [c["rank"] for c in cands]
        assert ranks == [1, 2, 3, 4, 5], (
            f"Rank incoherence in '{q['question_id']}': {ranks}"
        )

        for c in cands:
            missing = required_candidate_fields - set(c.keys())
            assert not missing, (
                f"Missing candidate fields in '{q['question_id']}': {missing}"
            )


def test_read_only_inspection_non_mutation():
    repo_root = Path(__file__).resolve().parent.parent
    calib_path = repo_root / "evaluation" / "results" / "retrieval_calibration.json"

    # Capture initial SHA256 and production settings
    hash_before = hashlib.sha256(calib_path.read_bytes()).hexdigest()
    top_k_before = settings.TOP_K
    threshold_before = settings.RELEVANCE_THRESHOLD

    # Perform read-only inspections
    test_phase03_calibration_artifact_integrity_and_sha256()
    test_phase09_experiment_artifact_relationship_validation()
    test_phase09_evidence_artifact_completeness_and_coherence()

    # Capture final SHA256 and production settings
    hash_after = hashlib.sha256(calib_path.read_bytes()).hexdigest()
    top_k_after = settings.TOP_K
    threshold_after = settings.RELEVANCE_THRESHOLD

    assert hash_before == hash_after, "Calibration JSON hash mutated during inspection!"
    assert top_k_before == top_k_after == 4, "settings.TOP_K mutated!"
    assert threshold_before == threshold_after == 0.67, (
        "settings.RELEVANCE_THRESHOLD mutated!"
    )
