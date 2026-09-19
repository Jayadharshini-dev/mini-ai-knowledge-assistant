# Evaluation & Testing Strategy

## 1. Evaluation Scope
This document describes the evaluation, testing, and threshold calibration methodology for the Mini AI Knowledge Assistant (Mini AKA).

Evaluation in Mini AKA is structured around four distinct automated evaluation boundaries:
- **Dense Retrieval Evaluation**: Measuring raw vector search recall and ranking quality (Hit@1, Hit@3, Recall@5, MRR) over grounded document passages.
- **Relevance Gate Calibration**: Data-driven optimization of cosine similarity thresholding to distinguish in-scope document queries from out-of-scope queries.
- **Out-of-Scope Abstention**: Verifying that unanswerable or out-of-domain inquiries are deterministically rejected with clean abstention responses.
- **Evidence & Citation Validation**: Mechanically validating that LLM answer claims are backed by retrieved document page numbers and chunk IDs.

> [!NOTE]
> **Formal Human Groundedness Evaluation**: Automated metrics, threshold calibration, and evidence validation were conducted on the benchmark dataset. However, formal human groundedness evaluation or LLM-as-judge scoring was **not** performed.

---

## 2. Evaluation Dataset
Evaluation uses a fixed 30-question benchmark dataset (`evaluation/dataset.yaml`) grounded in the three primary domain PDF documents:
1. `campus_resource_management.pdf`
2. `edge_computing_iot.pdf`
3. `predictive_maintenance_ml.pdf`

### Dataset Breakdown
- **Total Questions**: 30
- **In-Scope Questions**: 20 (queries with ground-truth document and page evidence pairs)
- **Out-of-Scope Questions**: 10 (queries deliberately outside the domain corpus, requiring abstention)

---

## 3. Retrieval Metrics
Retrieval quality is measured on the 20 in-scope queries using standard information retrieval metrics evaluated at $K=5$:

- **Hit@1**: Proportion of queries where at least one ground-truth evidence chunk is ranked at position 1.
  $$\text{Hit@1} = \frac{1}{|Q_{\text{in}}|} \sum_{q \in Q_{\text{in}}} \mathbb{I}(\text{rank}_{\text{first\_match}} \le 1)$$
- **Hit@3**: Proportion of queries where at least one ground-truth evidence chunk is retrieved within the top 3 results.
  $$\text{Hit@3} = \frac{1}{|Q_{\text{in}}|} \sum_{q \in Q_{\text{in}}} \mathbb{I}(\text{rank}_{\text{first\_match}} \le 3)$$
- **Recall@5**: Fraction of expected ground-truth evidence pairs present in the top 5 retrieved chunks.
  $$\text{Recall@5} = \frac{1}{|Q_{\text{in}}|} \sum_{q \in Q_{\text{in}}} \frac{|\text{Retrieved}_{5}(q) \cap \text{Expected}(q)|}{|\text{Expected}(q)|}$$
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of the first relevant chunk across in-scope queries.
  $$\text{MRR} = \frac{1}{|Q_{\text{in}}|} \sum_{q \in Q_{\text{in}}} \frac{1}{\text{rank}_{\text{first\_match}}(q)}$$

---

## 4. Threshold Calibration
To prevent hallucination on out-of-scope questions while preserving high coverage on in-scope questions, the relevance threshold was calibrated using a candidate threshold sweep from $0.05$ to $0.70$ across 66 evaluation points.

- **Historical Pre-calibration Threshold**: `0.32`
- **Calibrated Production Threshold**: `0.67`

### Relevance Gate Metrics
The relevance gate evaluates retrieved candidate chunks against similarity threshold $\tau$:
- **In-Scope Coverage**: Proportion of in-scope questions where top chunk similarity score $\ge \tau$ (decision = `proceed`).
- **Out-of-Scope Abstention Rate**: Proportion of out-of-scope questions where top chunk similarity score $< \tau$ (decision = `abstain`).
- **Gate F1 Score**: Harmonic mean of Gate Precision (abstention on out-of-scope) and Gate Recall (coverage on in-scope).
  $$\text{Gate F1} = 2 \cdot \frac{\text{Precision}_{\text{gate}} \cdot \text{Recall}_{\text{gate}}}{\text{Precision}_{\text{gate}} + \text{Recall}_{\text{gate}}}$$

At candidate threshold $\tau = 0.67$, the gate achieved the optimal balance of **100.0% out-of-scope abstention** and **90.0% in-scope coverage**, yielding a Gate F1 score of **0.9474**.

---

## 5. Production Baseline
The production retrieval architecture uses:
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Vector Normalization**: L2-normalized query and document vectors
- **Index Type**: FAISS `IndexFlatIP` (Cosine Similarity)
- **Production Top-K**: `4` (Runtime context injection)
- **Evaluation Candidates Top-K**: `5` (Offline evaluation & evidence log depth)
- **Relevance Threshold**: `0.67`

> [!NOTE]
> **Production Top-K vs. Evaluation Top-K**: At runtime, production RAG context assembly retrieves the top **`TOP_K = 4`** candidates above threshold (`RELEVANCE_THRESHOLD = 0.67`). Offline benchmark evaluation and evidence logging retrieve **`K = 5`** candidate chunks per query to calculate top-5 evaluation metrics (e.g., `Recall@5`).

### Measured Production Baseline Performance
Derived from authoritative evaluation artifacts (`retrieval_calibration.json` and `retrieval_experiments.json`):

| Metric | Measured Baseline Score |
| :--- | :---: |
| **Hit@1** | 0.8000 |
| **Hit@3** | 1.0000 |
| **Recall@5** | 1.0000 |
| **MRR** | 0.8833 |
| **In-Scope Coverage ($\tau=0.67$)** | 0.9000 (18 / 20) |
| **Out-of-Scope Abstention Rate ($\tau=0.67$)** | 1.0000 (10 / 10) |
| **Gate Precision** | 1.0000 |
| **Gate Recall** | 0.9000 |
| **Gate F1 Score** | **0.9474** |

---

## 6. Retrieval Ablations & Comparisons

Two controlled experiment configurations were evaluated alongside the baseline in Phase 09 without altering production settings:

### 1. Unnormalized Embedding Ablation (`unnormalized_embeddings_ablation`)
- **Configuration**: Raw unnormalized document and query vectors evaluated on a separate in-memory `IndexFlatIP(384)`.
- **Ranking Metrics**: Hit@1 = 0.8000, Hit@3 = 1.0000, Recall@5 = 1.0000, MRR = 0.8833.
- **Observed Score Range**: `[0.3592, 0.8679]`.
- **Gating Status**: Not Applicable (`gating_applicable: false`, `gating_metrics: null`).
- **Finding**: While unnormalized inner products preserve candidate ranking order on this corpus, unnormalized scores produce arbitrary vector magnitudes that cannot be gated against a fixed similarity threshold. L2-normalization is required to bound scores to the `[-1.0, 1.0]` cosine range.

### 2. Pre-calibration Threshold Comparison (`pre_calibration_threshold_comparison`)
- **Configuration**: Reuses production baseline retrieved candidates evaluated with pre-calibration threshold $\tau = 0.32$.
- **Ranking Metrics**: Identical to baseline by construction.
- **Gating Metrics**: Coverage = 1.0000 (20/20), Out-of-Scope Abstention = 0.0000 (0/10), Gate Precision = 0.6667, Gate F1 = **0.8000**.
- **Finding**: Setting the threshold to 0.32 fails to reject any of the 10 out-of-scope questions, causing a 0.0% abstention rate on invalid inquiries. Calibrating the threshold to 0.67 improves Gate F1 from 0.8000 to 0.9474.

---

## 7. Evidence & Evaluation Artifacts

All evaluation results are persisted in machine-readable JSON artifacts:
- **`evaluation/results/retrieval_calibration.json`**: Primary threshold calibration sweep data and frozen Phase 03 baseline metrics.
- **`evaluation/results/retrieval_experiments.json`**: Experiment comparison results for baseline, unnormalized ablation, and pre-calibration threshold comparison, including Phase 03 preservation record and SHA256 hashes.
- **`evaluation/results/retrieval_evidence.json`**: Per-question evidence records for all 30 evaluation queries detailing retrieved candidates ($K=5$), ranks, similarity scores, threshold decisions, and ground-truth evidence matching.
- **`docs/figures/retrieval_experiments.png`**: Metric comparison chart contrasting baseline, ablation, and threshold comparison scores.

---

## 8. Reproducibility & Safe Commands

To inspect or reproduce evaluation metrics safely without mutating historical artifacts or production vector index files, use the following read-only commands:

### Running Evaluation Experiments
```bash
python -m evaluation.experiments
```
*Note: Executes isolated in-memory experiment passes and asserts non-mutation of application settings and calibration files.*

### Running Automated Test Suite
```bash
python -m pytest
```

### Checking Lint & Formatting
```bash
python -m ruff check .
python -m ruff format --check .
```

> [!CAUTION]
> **Destructive Commands Warning**: Do **NOT** run `python scripts/build_index.py --force` or `python -m evaluation.calibrate` during routine evaluation inspection. `build_index --force` rebuilds vector index files, and `evaluation.calibrate` re-executes threshold selection.

---

## 9. Automated Test Suite Architecture

The project test suite provides multi-layered coverage across all application components:

- **Ingestion & Processing (`test_loader.py`, `test_cleaner.py`, `test_chunker.py`)**: Tests PDF text extraction, page cleaning, BGE token counting, and chunk generation.
- **Embedder & Vector Store (`test_embeddings.py`, `test_vector_store.py`)**: Tests BGE query prefixing, float32 array shapes, FAISS `IndexFlatIP` search, atomic saving, reload, and manifest mismatch guards.
- **Retrieval & Relevance (`test_retriever.py`, `test_relevance.py`, `test_metrics.py`)**: Tests dense retrieval execution, cosine threshold filtering, and IR metric calculations.
- **Generation & Citation (`test_prompt.py`, `test_providers.py`, `test_context_builder.py`, `test_citation_validator.py`, `test_generation_abstention.py`)**: Tests prompt construction, provider interfaces, context string building, citation extraction/validation, and hallucination prevention.
- **RAG Pipeline & State (`test_pipeline_happy.py`, `test_pipeline_contract.py`, `test_pipeline_abstention.py`, `test_pipeline_degraded.py`, `test_pipeline_errors.py`, `test_pipeline_integration.py`, `test_events.py`)**: Tests end-to-end RAG state transitions, SSE trace event streams, degraded provider fallbacks, and failure handling.
- **API & Frontend Contracts (`test_api.py`, `test_health.py`, `test_frontend_contracts.py`)**: Tests REST/SSE HTTP endpoints, non-leaking exception handling, and SSE trace payload compatibility with frontend contracts.
- **Experiments & Artifact Integrity (`test_retrieval_experiments.py`, `test_evaluation_artifacts.py`)**: Tests Phase 09 experiment harness, artifact schemas, per-question evidence completeness, rank coherence, SHA256 file preservation, and read-only disk safety.

---

## 10. Empirical Limitations

1. **Evaluation Dataset Size**: Evaluation is conducted on a 30-question benchmark dataset across 3 technical domain PDFs.
2. **CPU Execution**: Vector embedding and FAISS search are benchmarked and executed on CPU hardware.
3. **No Reranker / Hybrid Search**: Retrieval uses single-stage dense vector search without BM25 hybrid search or cross-encoder reranking.
4. **Un-evaluated Capabilities**: Production evaluation does not benchmark multi-user concurrent load performance, OCR for scanned documents, or token-by-token streaming latency.

---

## 11. Technical Provenance & Integrity Notes

- **Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Vector Dimension**: `384`
- **Similarity Index**: FAISS `IndexFlatIP` over L2-normalized vectors
- **Production Configuration**: `TOP_K = 4`, `RELEVANCE_THRESHOLD = 0.67`
- **Historical Configured Threshold**: `0.32`
- **Phase 03 Calibration Artifact SHA256**: `c1b1d00a8826bb2c70d637eb9fdc4845104a106292a3b084daa3d3c8c50b155d`
