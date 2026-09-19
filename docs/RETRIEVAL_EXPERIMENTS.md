# Retrieval Experiments & Evidence Layer Report

## 1. Overview & Purpose
This report documents the Phase 09 retrieval experiments conducted on the Mini AI Knowledge Assistant evaluation benchmark dataset (`evaluation/dataset.yaml`, 30 questions). The objective is to provide empirical, reproducible evidence answering why the production dense retriever configuration is structured as deployed.

The production configuration remains completely unchanged:
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Normalization**: L2-normalized query and document vectors
- **Index Type**: FAISS `IndexFlatIP` (Cosine Similarity)
- **Top-K**: `4` (Evaluation K = `5`)
- **Relevance Threshold**: `0.67`

---

## 2. Experimental Configurations

Three controlled configurations were evaluated under strict isolation without mutating production settings:

1. **Production Baseline (`production_baseline`)**:
   - L2-normalized BGE embeddings.
   - FAISS `IndexFlatIP` index over cosine similarity.
   - Calibrated relevance threshold = `0.67`.

2. **Unnormalized Embedding Ablation (`unnormalized_embeddings_ablation`)**:
   - Same BGE model singleton.
   - Raw unnormalized document vectors and query vectors.
   - Separate in-memory FAISS `IndexFlatIP(384)` calculating raw inner products.
   - Gating metrics explicitly set to **Not Applicable** (`gating_applicable: false`).

3. **Pre-calibration Threshold Comparison (`pre_calibration_threshold_comparison`)**:
   - Identical retrieved candidate chunks as the production baseline.
   - Uncalibrated default threshold = `0.32`.
   - Gating calibration comparison (`comparison_type: "gating_calibration"`).

---

## 3. Metric Comparison Table

| Metric | Production Baseline (`0.67`) | Unnormalized Ablation | Pre-calibration Threshold (`0.32`) |
| :--- | :---: | :---: | :---: |
| **Hit@1** | 0.8 | 0.8 | 0.8 |
| **Hit@3** | 1.0 | 1.0 | 1.0 |
| **Recall@5** | 1.0 | 1.0 | 1.0 |
| **MRR** | 0.8833 | 0.8833 | 0.8833 |
| **In-Scope Coverage** | 0.9 | N/A | 1.0 |
| **Out-of-Scope Abstention Rate** | 1.0 | N/A | 0.0 |
| **Gate Precision** | 1.0 | N/A | 0.6667 |
| **Gate Recall** | 0.9 | N/A | 1.0 |
| **Gate F1 Score** | 0.9474 | N/A | 0.8 |
| **Observed Raw Score Range** | `[0.3705, 0.8872]` | `[0.3592, 0.8679]` | `[0.3705, 0.8872]` |

---

## 4. Empirical Observations & Findings

1. **Impact of L2-Normalization**:
   - On the current 30-question evaluation dataset, both normalized and unnormalized embeddings achieved identical top-K ranking scores (Hit@1 = 0.8, Hit@3 = 1.0, Recall@5 = 1.0, MRR = 0.8833).
   - However, unnormalized dot products produce arbitrary raw vector magnitudes ranging from `0.3592` to `0.8679` depending on chunk text length and vector norm.
   - Without L2-normalization, dot product scores are unbounded and cannot be gated against a fixed similarity threshold. L2-normalization is required to bound similarity scores to the `[-1.0, 1.0]` cosine range.

2. **Impact of Threshold Calibration (`0.67` vs `0.32`)**:
   - Pre-calibration threshold `0.32` resulted in an **out-of-scope abstention rate of 0.0%**, failing to reject any of the 10 out-of-scope questions. This yielded a Gate F1 score of **0.8000** (Precision = 0.6667, Recall = 1.0000).
   - Production calibrated threshold `0.67` achieved **100.0% out-of-scope abstention** while maintaining **90.0% in-scope coverage**, yielding a Gate F1 score of **0.9474**.

---

## 5. Evidence Artifact & Per-Question Inspection
Detailed evidence records for all 30 evaluation questions are persisted in `evaluation/results/retrieval_evidence.json`. For each question, all K=5 retrieved candidate chunks, text previews, rank, exact cosine scores, threshold decisions, and ground-truth evidence matches are recorded.

---

## 6. Reproducibility & Phase 03 Preservation
- **Phase 03 Artifact SHA256**: `c1b1d00a8826bb2c70d637eb9fdc4845104a106292a3b084daa3d3c8c50b155d`
- **Phase 03 Base Metrics Preserved**: `hit_at_1 = 0.8`, `hit_at_3 = 1.0`, `recall_at_5 = 1.0`, `mrr = 0.8833`
- **Application Settings Unmutated**: `TOP_K = 4`, `RELEVANCE_THRESHOLD = 0.67`
- **Index Status**: Production index was rebuilt during implementation verification before the final experiment run.
