# Member 2 (P2) — Social Media Brand Sentiment Analysis Deliverables

## Role & Assignment Overview
- **Role**: Member 2 (P2) — Sentiment Analysis, Transformer Fine-Tuning & Inference Pipeline
- **Target Branch**: `P2` (Base commit: `9dd7c43` on remote `team/P2`)
- **Core Assignment**:
  - Fine-tune and integrate sequence classification models (**DistilBERT v1** and **DistilBERT v2**) with softmax confidence scores and offline execution.
  - Implement a model factory prioritizing DistilBERT v2 with fallback to a classical TF-IDF + Logistic Regression baseline.
  - Deliver a batch CSV inference script (`scripts/label_csv.py`) that auto-detects comment columns, preserves metadata/IDs, and exports segregated `positive.csv`, `negative.csv`, and `neutral.csv` files.
  - Maintain an automated test suite verifying model loading, label mapping, factory selection, API endpoints, and batch exports (81 passing tests on the clean `team/P2` branch; 88 passing tests on `p2-p3-integration`).

---

## 1. Verified Deliverables & Component Attribution

### A. Sentiment Model Architecture & Factory
| File | Scope & Technical Implementation |
|---|---|
| `src/sentiment/distilbert_analyzer.py` | DistilBERT sequence classification analyzer. Handles offline model path resolution (`_resolve_model_path`) preferring v2 (`models/distilbert/gfg_distilbert_sentiment_v2`) over v1, computes softmax confidence scores, and enforces verified 3-class label mapping (`0=negative`, `1=neutral`, `2=positive`). |
| `src/sentiment/factory.py` | Model factory selecting `DistilBertSentimentAnalyzer` by default, preferring v2 when present, and falling back gracefully to TF-IDF baseline if transformer artifacts are unavailable. |
| `src/sentiment/analyzer.py` | Classical sentiment analyzer using TF-IDF feature extraction and Logistic Regression (baseline engine). |
| `src/sentiment/__init__.py` | Unified module export interface. |

### B. Batch Inference & Training Pipelines
| File | Scope & Technical Implementation |
|---|---|
| `scripts/label_csv.py` | Batch CSV inference tool. Features candidate column detection (`comment`, `text`, `content`, `body`, `review`), ID preservation, batched inference, and output segregation into `positive.csv`, `negative.csv`, and `neutral.csv`. |
| `scripts/train_distilbert.py` | Hugging Face Trainer fine-tuning pipeline supporting DistilBERT v1 and v2, class-weighted cross-entropy for imbalanced classes, stratified splitting, and evaluation metric logging (Macro F1, precision, recall). |
| `scripts/prepare_distilbert_v2_data.py` | DistilBERT v2 dataset preparation script implementing stratified sampling and text normalization. |
| `scripts/validate_distilbert_artifact.py` | Standalone validator checking transformer artifacts (`model.safetensors`, `config.json`, tokenizer files) prior to deployment. |
| `scripts/train_sentiment.py` | Baseline TF-IDF + Logistic Regression training and evaluation script. |

### C. Foundational Architecture & Integration Utilities
| Component | Files | Description |
|---|---|---|
| **Pipeline** | `src/pipeline.py` | Pipeline orchestrating text cleaning, sentiment inference, and negative complaint triage. |
| **Preprocessing** | `src/preprocessing/cleaner.py`<br>`src/preprocessing/validator.py` | Text cleaning (regex, URLs, emojis, HTML entities) and input validation schemas. |
| **Data Ingestion** | `src/data/loader.py` | CSV/JSON ingestion utilities with validation. |
| **REST API** | `src/api/app.py`<br>`src/api/schemas.py` | FastAPI application exposing `/health`, `/model/info`, `/predict`, `/predict/batch`, and `/clusters`. |
| **Startup Scripts** | `main.py`<br>`run_dashboard.py` | Root-level entrypoints for backend server and dashboard startup. |

### D. Optional Integration-Branch Work (`p2-p3-integration`)
An experimental dataset-builder script (`scripts/build_150k_multi_source_dataset.py`) is maintained separately on `team/p2-p3-integration` for high-throughput pipeline testing.
- **Purpose**: Generates an unlabelled 150,000-record inference corpus by aggregating public social media posts and review records from Hugging Face and Kaggle to stress-test batch throughput and downstream clustering.
- **Leakage Filtering**: The script incorporates deduplication against known benchmark splits (`data/benchmark/`), filtering out 56,488 rows that overlapped with DistilBERT v2 training data.
- **Evaluation Integrity**: This 150K corpus is strictly an unlabelled inference corpus. No evaluation metrics or benchmarks are claimed from it. All reported model performance metrics remain anchored to the held-out test split recorded in `models/distilbert/gfg_distilbert_sentiment_v2/evaluation_report.json`.

---

## 2. Commit History on `team/P2`

The commits added on the `P2` branch ahead of the initial repository baseline (`5197fb0`) are recorded in Git under Pranav's author identity. Git commit metadata reflects the repository's logged revision history rather than serving as independent proof of code authorship; attribution is corroborated by the verified implementation diffs and test suites documented above:

| Commit ID | Recorded Author Identity | Date | Summary |
|---|---|---|---|
| `9dd7c43` | Pranav | 2026-10-12 | `feat: integrate fine-tuned DistilBERT v2 sentiment model` |
| `545b6fc` | Pranav | 2026-10-11 | `chore: reconcile team/P2 baseline with GFG probation project` |
| `9160266` | Pranav | 2026-10-11 | `feat: add sentiment CSV export and training scripts` |
| `2a27701` | Pranav | 2026-10-05 | `feat: integrate DistilBERT sentiment model` |
| `13aa1a8` | Pranav | 2026-10-05 | `feat: implement social media brand sentiment & complaint clusterer pipeline...` |
| `e183065` | Pranav | 2026-10-05 | `feat: add csv loading and preprocessing pipeline` |
| `754f6e2` | Pranav | 2026-10-05 | `feat: add text cleaning function` |
| `47f564f` | Pranav | 2026-10-05 | `first commit` |

---

## 3. Explicit Teammate Attribution Boundaries

To preserve strict separation of responsibilities:
- **Member 3 (Complaint Clustering — Mayank)**:
  - Preserved on `team/mayank` (`b1d0745`) and integration branch `team/p2-p3-integration`.
  - Authored files: `src/clustering/embeddings.py`, `scripts/evaluate_embeddings_clustering.py`, `scripts/compare_clustering_methods.py`, `scripts/export_clustered_complaints.py`, `results/`, `data/processed/clustered_comments.csv`, `pytest.ini`, 7 embedding clustering tests in `tests/test_clustering.py`, and Section 17 of `README.md`.
- **Member 4 (Dashboard Integration)**:
  - Preserved on `team/feature/p4` and integration branch `team/p2-p3-integration`.
- **Consolidated Integration**:
  - Full multi-role pipeline integration (Sentiment + Embedding Clustering + Dashboard) is maintained on `team/p2-p3-integration`.

---

## 4. Test Suite Execution & Attribution Dissection

The repository contains two verified test baselines depending on the branch context:

| Test Module | Pure P2 Branch (`team/P2` @ `9dd7c43`) | Consolidated Integration Branch (`p2-p3-integration` @ `ed4e724`) | Module Scope & Test Target |
|---|---|---|---|
| `tests/test_api.py` | **10 passed** | **10 passed** | FastAPI health, info, predict, and batch routes |
| `tests/test_clustering.py` | **5 passed** | **12 passed** (+7) | TF-IDF KMeans baseline (5 tests) vs. Embeddings/DBSCAN (7 tests by Mayank) |
| `tests/test_data_loader.py` | **8 passed** | **8 passed** | CSV/JSON ingestion, validation, and format handling |
| `tests/test_data_preparation.py` | **8 passed** | **8 passed** | DistilBERT v2 dataset split balance & stratification |
| `tests/test_distilbert.py` | **12 passed** | **12 passed** | DistilBERT v1 & v2 offline loading, tokenizer, and prediction |
| `tests/test_label_csv.py` | **12 passed** | **12 passed** | Column auto-detection, batch inference, and CSV segregation |
| `tests/test_pipeline.py` | **4 passed** | **4 passed** | End-to-end sentiment + clustering pipeline execution |
| `tests/test_preprocessing.py` | **10 passed** | **10 passed** | Text cleaner regex (URLs, emojis, HTML) and input validator |
| `tests/test_sentiment.py` | **6 passed** | **6 passed** | Classical TF-IDF baseline analyzer training & prediction |
| `tests/test_sentiment_factory.py` | **6 passed** | **6 passed** | Model factory selection, version preference, and fallback logic |
| **Total Test Count** | **81 passed** (47.08s) | **88 passed** (59.56s) | **100% pass rate in both configurations** |

*Summary*:
- **Clean P2 Branch**: Running `python -m pytest` on `team/P2` executes **81 tests**, covering sentiment analysis models, batch labeling, API endpoints, data utilities, and baseline clustering.
- **Integration Branch**: Running `python -m pytest` on `p2-p3-integration` executes **88 tests** (the 81 baseline tests plus 7 embedding-clustering tests contributed by Mayank).

---

## 5. Evaluator Verification Commands

### 1. Execute Automated Test Suite:
```powershell
python -m pytest
```

### 2. Run Batch CSV Inference with DistilBERT v2:
```powershell
python scripts/label_csv.py --input-file data/raw/comments_batch_01.csv --output-dir data/processed
```
*(Automatically selects DistilBERT v2, preserves original IDs, and outputs `data/processed/positive.csv`, `negative.csv`, and `neutral.csv`)*

### 3. Verify Model Factory Selection in Python:
```python
from src.sentiment import get_sentiment_analyzer

analyzer = get_sentiment_analyzer()
print(analyzer.version)       # Outputs: 'v2'
print(analyzer.model_path)     # Outputs: 'models/distilbert/gfg_distilbert_sentiment_v2'
print(analyzer.predict("I absolutely love this product! Best purchase ever."))
# Outputs: {'sentiment': 'positive', 'confidence': 0.99...}
```
