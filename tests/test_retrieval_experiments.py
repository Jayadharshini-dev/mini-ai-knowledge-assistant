from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import settings
from evaluation.experiments import run_experiments, verify_phase03_calibration


def test_phase09_retrieval_experiments_and_evidence():
    repo_root = Path(__file__).resolve().parent.parent
    calib_path = repo_root / "evaluation" / "results" / "retrieval_calibration.json"
    exp_json_path = repo_root / "evaluation" / "results" / "retrieval_experiments.json"
    evidence_json_path = (
        repo_root / "evaluation" / "results" / "retrieval_evidence.json"
    )

    # 1. Run experiment harness
    exp_payload = run_experiments()
    assert exp_json_path.exists()

    # 2. Verify global settings are not mutated
    assert settings.TOP_K == 4
    assert settings.RELEVANCE_THRESHOLD == 0.67

    # 3. Verify exactly 3 experiments exist
    exps = exp_payload["experiments"]
    assert len(exps) == 3
    assert "production_baseline" in exps
    assert "unnormalized_embeddings_ablation" in exps
    assert "pre_calibration_threshold_comparison" in exps

    # 4. Verify Experiment 1 (Production Baseline) configuration and metrics
    prod = exps["production_baseline"]
    assert prod["configuration"]["threshold"] == 0.67
    assert prod["metrics"]["hit_at_1"] == 0.8
    assert prod["metrics"]["hit_at_3"] == 1.0
    assert prod["metrics"]["recall_at_5"] == 1.0
    assert prod["metrics"]["mrr"] == 0.8833
    assert prod["metrics"]["gating_applicable"] is True

    # 5. Verify Experiment 2 (Unnormalized Ablation) configuration and metrics
    unnorm = exps["unnormalized_embeddings_ablation"]
    assert unnorm["configuration"]["normalized_embeddings"] is False
    assert unnorm["metrics"]["gating_applicable"] is False
    assert unnorm["metrics"]["gating_metrics"] is None
    assert "observed_raw_score_range" in unnorm["metrics"]

    # 6. Verify Experiment 3 (Pre-calibration Threshold) configuration and metrics
    thresh032 = exps["pre_calibration_threshold_comparison"]
    assert thresh032["configuration"]["threshold"] == 0.32
    assert thresh032["metrics"]["comparison_type"] == "gating_calibration"
    assert thresh032["metrics"]["shares_retrieval_with"] == "production_baseline"
    assert thresh032["metrics"]["gating_applicable"] is True

    # 7. Verify Phase 03 preservation record
    preservation = exp_payload["phase03_preservation"]
    assert preservation["configured_threshold"] == 0.32
    assert preservation["calibrated_threshold"] == 0.67
    assert preservation["base_metrics"]["hit_at_1"] == 0.8
    assert preservation["base_metrics"]["hit_at_3"] == 1.0
    assert preservation["base_metrics"]["recall_at_5"] == 1.0
    assert preservation["base_metrics"]["mrr"] == 0.8833
    assert preservation["metrics_match"] is True
    assert preservation["calibration_file_rewritten"] is False

    # 8. Verify reproducibility metadata
    repro = exp_payload["reproducibility"]
    assert isinstance(repro["production_index_rebuilt"], bool)
    assert repro["application_settings_unmutated"] is True

    # 9. Verify Phase 03 calibration file on disk was not modified
    calib_data, _ = verify_phase03_calibration(calib_path)
    assert calib_data["metadata"]["configured_threshold"] == 0.32
    assert calib_data["metadata"]["calibrated_threshold"] == 0.67

    # 10. Verify Evidence Artifact (30 questions, K=5 candidates each)
    assert evidence_json_path.exists()
    with open(evidence_json_path, encoding="utf-8") as f:
        evidence_data = json.load(f)

    assert len(evidence_data) == 30
    for q in evidence_data:
        assert "question_id" in q
        assert "question" in q
        assert "in_scope" in q
        assert "expected_evidence" in q
        assert "hit_at_1" in q
        assert "reciprocal_rank" in q
        assert "decision" in q
        assert "selected_count" in q
        assert "retrieved_candidates" in q

        cands = q["retrieved_candidates"]
        assert len(cands) == 5
        for c in cands:
            assert "rank" in c
            assert "chunk_id" in c
            assert "document" in c
            assert "page" in c
            assert "chunk_index" in c
            assert "score" in c
            assert "above_threshold" in c
            assert "is_relevant" in c
            assert "text_preview" in c
            assert "text_truncated" in c


def test_calibrate_script_preserves_historical_threshold_provenance():
    repo_root = Path(__file__).resolve().parent.parent
    calibrate_py_path = repo_root / "evaluation" / "calibrate.py"
    content = calibrate_py_path.read_text(encoding="utf-8")

    assert '"configured_threshold": 0.32,' in content
