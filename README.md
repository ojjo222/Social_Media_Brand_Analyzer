# Social Media Brand Sentiment & Complaint Clusterer

A production-quality Natural Language Processing (NLP) system designed to analyze brand-related comments from social media platforms (Twitter, Instagram, Facebook, Reddit), classify sentiment, identify negative feedback, automatically cluster complaints into actionable business themes, and present insights through a FastAPI backend and a Streamlit interactive dashboard.

The system features fine-tuned **DistilBERT** as its primary transformer sentiment model (+7.85 percentage points Macro F1 improvement over classical baseline) alongside an optimized **TF-IDF + Logistic Regression** fallback engine.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Installation](#installation)
5. [Environment Setup](#environment-setup)
6. [Dataset Format](#dataset-format)
7. [Training & Benchmark Instructions](#training--benchmark-instructions)
8. [Model Details & Artifact Strategy](#model-details--artifact-strategy)
9. [Final Benchmark Evaluation Results](#final-benchmark-evaluation-results)
10. [Running the API](#running-the-api)
11. [Running the Dashboard](#running-the-dashboard)
12. [Example API Requests & Responses](#example-api-requests--responses)
13. [Testing](#testing)
14. [Limitations](#limitations)
15. [Future Improvements](#future-improvements)

---

## 1. Project Overview

Brands receive thousands of customer mentions, reviews, and inquiries across various social media platforms daily. Manually sorting through these comments to identify dissatisfied customers and categorize operational issues is slow and prone to errors.

This project delivers an end-to-end automated pipeline that:
- **Ingests & Validates** social media text and metadata across platforms and brands.
- **Preprocesses & Cleans** informal text (handling URLs, mentions, hashtags, character repetitions, HTML entities, and emojis).
- **Classifies Sentiment** via a fine-tuned DistilBERT transformer (or TF-IDF baseline) into Positive, Neutral, or Negative with calibrated confidence scores.
- **Filters Negative Feedback** for high-priority operational triage.
- **Clusters Complaints** using unsupervised learning to group similar issues together.
- **Extracts Themes & Labels** representing real operational problems (e.g., Delivery Problems, Refund Issues, App Bugs, Customer Support, Billing & Overcharging, Product Quality).
- **Serves Predictions** via high-performance FastAPI REST endpoints.
- **Visualizes Metrics** through an interactive, multi-tab Streamlit dashboard.

---

## 2. Architecture

```text
[ Social Media Comments ]
           │
           ▼
┌─────────────────────────────────┐
│     Data Ingestion & Loader     │ (CSV, JSON, Python Dicts)
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Data Validation & Cleaning    │ (Regex, HTML unescape, Tokenizer)
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Sentiment Analysis Engine     │ ──► Primary: DistilBERT (83.97% Macro F1)
│   (Configurable via Env Var)    │ ──► Fallback: TF-IDF + Logistic Regression (76.12% Macro F1)
└─────────────────────────────────┘
           │
           ├─► [Positive / Neutral] ──────────────────────────────────────────┐
           │                                                                 │
           ▼ (Negative comments only)                                        │
┌─────────────────────────────────┐                                         │
│   Complaint Clusterer (K-Means) │                                         │
└─────────────────────────────────┘                                         │
           │                                                                 │
           ▼                                                                 │
┌─────────────────────────────────┐                                         │
│    Theme & Keyword Extraction   │                                         │
│    (Cluster ID, Terms, Theme)   │                                         │
└─────────────────────────────────┘                                         │
           │                                                                 │
           ▼                                                                 │
┌────────────────────────────────────────────────────────────────────────┐   │
│                 Aggregated Prediction & Insights                       │◄──┘
└────────────────────────────────────────────────────────────────────────┘
           │
     ┌─────┴──────────────┐
     ▼                    ▼
┌──────────────┐   ┌──────────────┐
│  FastAPI App │   │  Streamlit   │
│  (:8000)     │   │  Dashboard   │
└──────────────┘   └──────────────┘
```

---

## 3. Folder Structure

```text
GFG_PROJ/
├── .env.example                     # Environment template configuration
├── .gitignore                        # Git exclusion rules (ignores models/)
├── main.py                           # Application entry point (runs FastAPI server)
├── README.md                         # Comprehensive project documentation
├── requirements.txt                  # Python dependencies
├── run_dashboard.py                  # Streamlit launcher script
├── data/
│   ├── benchmark/                    # Real-world benchmark dataset splits
│   │   ├── train.csv                 # 59,780 real comments
│   │   ├── validation.csv
│   │   └── test.csv                  # 7,473 held-out test comments
│   ├── raw/                          # Raw comment batches
│   ├── processed/                    # Cleaned and prepared data
│   └── synthetic/                    # Generated synthetic development datasets
│       ├── comments.csv              # Full dataset for local testing
│       └── sentiment_train.csv       # Dedicated labeled sentiment dataset
├── models/                           # Serialized model artifacts (git-ignored)
│   ├── clustering/                   # K-Means clustering artifacts
│   │   ├── cluster_labels.json
│   │   ├── cluster_themes.json
│   │   ├── kmeans.joblib
│   │   ├── metrics.json
│   │   └── vectorizer.joblib
│   ├── distilbert/                   # Fine-tuned DistilBERT transformer
│   │   └── gfg_distilbert_sentiment/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   └── sentiment/                    # TF-IDF fallback artifacts
│       ├── metrics.json
│       └── pipeline.joblib
├── scripts/                          # Offline training & validation scripts
│   ├── generate_synthetic_data.py    # Generates synthetic brand social media data
│   ├── train_clustering.py           # Unsupervised K-Means clustering optimization
│   ├── train_sentiment.py            # Supervised TF-IDF + Logistic Regression training
│   └── validate_distilbert_artifact.py # Isolated verification for DistilBERT
├── src/                              # Core application source code
│   ├── __init__.py
│   ├── config.py                     # Centralized settings and model selection
│   ├── pipeline.py                   # Unified end-to-end inference pipeline
│   ├── api/                          # FastAPI web application
│   │   ├── __init__.py
│   │   ├── app.py                    # REST API routes and lifespan management
│   │   └── schemas.py                # Pydantic request and response schemas
│   ├── clustering/                   # Complaint clustering module
│   │   ├── __init__.py
│   │   └── clusterer.py              # ComplaintClusterer implementation
│   ├── dashboard/                    # Streamlit analytical dashboard
│   │   └── app.py                    # Multi-tab visualization app
│   ├── data/                         # Data loading utilities
│   │   ├── __init__.py
│   │   └── loader.py                 # File and dictionary loaders with error checks
│   ├── preprocessing/                # Text transformation & validation
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Text normalization and ML tokenization
│   │   └── validator.py              # Schema and input integrity validators
│   └── sentiment/                    # Sentiment classification module
│       ├── __init__.py
│       ├── analyzer.py               # TF-IDF SentimentAnalyzer implementation
│       ├── distilbert_analyzer.py    # DistilBertSentimentAnalyzer implementation
│       └── factory.py                # Model selection factory & fallback logic
└── tests/                            # Unit and integration test suite (57 tests)
    ├── __init__.py
    ├── test_api.py                   # API routes and validation test cases
    ├── test_clustering.py            # Clustering predictions and edge cases
    ├── test_data_loader.py           # CSV/JSON file reading and error handling
    ├── test_distilbert.py            # DistilBERT unit tests and device fallback
    ├── test_pipeline.py              # End-to-end pipeline execution tests
    ├── test_preprocessing.py         # Text cleaning and validator tests
    ├── test_sentiment.py             # TF-IDF baseline unit tests
    └── test_sentiment_factory.py     # Model switching and fallback tests
```

---

## 4. Installation

Ensure you have **Python 3.10+** installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavgrv7981/GFG_PROJ.git
   cd GFG_PROJ
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Environment Setup

Copy `.env.example` to create your local `.env` configuration file:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

Default settings in `.env`:
```ini
API_HOST=0.0.0.0
API_PORT=8000
MODEL_DIR=models
DATA_DIR=data
LOG_LEVEL=INFO
SENTIMENT_MODEL=distilbert
```

### Model Switching
The sentiment engine can be toggled via the `SENTIMENT_MODEL` environment variable:
- `SENTIMENT_MODEL=distilbert` (default): Uses the fine-tuned transformer. If artifacts are missing, it automatically logs a warning and falls back to TF-IDF.
- `SENTIMENT_MODEL=tfidf`: Explicitly uses the lightweight classical TF-IDF model.

---

## 6. Dataset Format

### Real-World Benchmark Dataset (`data/benchmark/`)
The benchmark datasets were extracted and split from real-world comments:
- `train.csv`: 59,780 labeled real-world comments used for model training.
- `validation.csv`: Validation split for hyperparameter tuning.
- `test.csv`: 7,473 held-out real-world comments reserved strictly for final benchmark evaluation.

### Synthetic Development Dataset (`data/synthetic/comments.csv`)
Used for local development, CI testing, and dashboard demo runs:

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique comment identifier |
| `text` | string | Raw social media post content |
| `platform` | string | Platform name (`twitter`, `instagram`, `facebook`, `reddit`) |
| `brand` | string | Target brand (`Amazon`, `Flipkart`, `Swiggy`, `Zomato`, `PhonePe`, `Paytm`) |
| `sentiment` | string | Label (`positive`, `neutral`, `negative`) |
| `complaint_category`| string | Operational category for negative posts |

---

## 7. Training & Benchmark Instructions

### DistilBERT Fine-Tuning Setup
The DistilBERT model was fine-tuned in Google Colab with the following hyperparameters:
- **Base Architecture**: `distilbert-base-uncased`
- **Training Set**: 59,780 real-world comments
- **Classes**: 3 classes (`0: negative`, `1: neutral`, `2: positive`)
- **Epochs**: 2
- **Batch Size**: 32
- **Learning Rate**: 2e-5 (AdamW with linear schedule)
- **Precision**: FP16 mixed precision
- **Hardware**: NVIDIA Tesla T4 GPU

### TF-IDF Baseline Training
To retrain the classical baseline on synthetic or local data:
```bash
python scripts/train_sentiment.py
```

### Complaint Clustering Training
To optimize and train the unsupervised K-Means clustering model:
```bash
python scripts/train_clustering.py
```

---

## 8. Model Details & Artifact Strategy

### DistilBERT Model (`models/distilbert/`)
- **Tokenizer**: BertTokenizer / WordPiece tokenizer via `tokenizer.json`.
- **Weights**: Serialized as Hugging Face `model.safetensors` (~255 MB).
- **Inference Mode**: Evaluated in `torch.no_grad()` mode with dynamic CPU/CUDA device detection.
- **Preprocessing**: Light cleaning via `clean_text_for_transformer` (strips URLs and unescapes HTML entities, while **preserving punctuation, word order, and stopwords** required by the transformer).

### Model Artifact Strategy
- **Git Exclusion**: `models/` is explicitly listed in `.gitignore` to prevent committing heavy binary weights (~255 MB) into standard Git history.
- **Local Artifact Placement**:
  Place the extracted model files into `models/distilbert/` (or `models/distilbert/gfg_distilbert_sentiment/`):
  ```text
  models/distilbert/gfg_distilbert_sentiment/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── training_args.bin
  ```
- **Fallback Guarantee**: If the DistilBERT directory is missing or unreadable, the system logs a clean warning and falls back to `SentimentAnalyzer` (TF-IDF).

---

## 9. Final Benchmark Evaluation Results

The final sentiment models were evaluated on the **7,473 held-out real-world comments** (`data/benchmark/test.csv`).

### Final Comparison Benchmark

| Model | Architecture | Training Comments | Test Comments | Test Accuracy | Test Macro F1 | Improvement |
|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF + Logistic Regression | 59,780 | 7,473 | 76.84% | **76.12%** | — |
| **New Model** | **DistilBERT (`distilbert-base-uncased`)** | 59,780 | 7,473 | **84.38%** | **83.97%** | **+7.85% F1** |

### DistilBERT Detailed Metrics (on 7,473 Held-Out Comments)
- **Accuracy**: **84.38%**
- **Macro Precision**: **84.00%**
- **Macro Recall**: **83.96%**
- **Macro F1-Score**: **83.97%**
- **Net Gain**: **+7.85 percentage points** over the TF-IDF baseline.

### Complaint Clustering Metrics (`models/clustering/metrics.json`)
Evaluated across $k=3$ through $k=10$ using the Silhouette Coefficient:
- **Optimal $k$**: 10 (Silhouette score: `0.3384`)
- **Extracted Themes**:
  - *Delivery Problems*
  - *Customer Support*
  - *Billing & Overcharging*
  - *Refund Issues*
  - *App & Technical Bugs*
  - *Product Quality*

---

## 10. Running the API

Start the FastAPI application via `main.py` or `uvicorn`:

```bash
# Option 1: Via main script
python main.py

# Option 2: Via Uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 11. Running the Dashboard

Launch the Streamlit dashboard in a separate terminal:

```bash
# Option 1: Using the root launcher
python run_dashboard.py

# Option 2: Using Streamlit CLI directly
streamlit run src/dashboard/app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

The dashboard sidebar indicates whether the backend is connected and displays the currently active model (`DistilBertSentimentAnalyzer` or `SentimentAnalyzer`).

---

## 12. Example API Requests & Responses

### Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```
**Response:**
```json
{
  "sentiment_model_loaded": true,
  "clustering_model_loaded": true,
  "sentiment_model_type": "DistilBertSentimentAnalyzer",
  "clustering_model_type": "ComplaintClusterer"
}
```

### Single Comment Prediction (Positive)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Super fast delivery from @Amazon, loved it! #awesome", "platform": "twitter", "brand": "Amazon"}'
```
**Response:**
```json
{
  "original_text": "Super fast delivery from @Amazon, loved it! #awesome",
  "cleaned_text": "super fast delivery from loved it",
  "sentiment": "positive",
  "sentiment_confidence": 0.984,
  "is_negative": false,
  "complaint_cluster": null,
  "complaint_category": null,
  "platform": "twitter",
  "brand": "Amazon"
}
```

### Single Comment Prediction (Negative with Clustering)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Charged twice on checkout and customer support is unresponsive! @PhonePe", "platform": "reddit", "brand": "PhonePe"}'
```
**Response:**
```json
{
  "original_text": "Charged twice on checkout and customer support is unresponsive! @PhonePe",
  "cleaned_text": "charged twice on checkout and customer support is unresponsive",
  "sentiment": "negative",
  "sentiment_confidence": 0.978,
  "is_negative": true,
  "complaint_cluster": 4,
  "complaint_category": "Customer Support",
  "platform": "reddit",
  "brand": "PhonePe"
}
```

---

## 13. Testing

The project maintains an automated test suite with **69 passing unit and integration tests**:

Run the full test suite:
```bash
python -m pytest -v
```

### Test Suite Structure:
- `tests/test_label_csv.py`: 12 tests covering CSV validation, ID/comment auto-detection, null/empty comment safety, sentiment segregation, schema uniformity, and row invariants.
- `tests/test_distilbert.py`: 10 tests covering transformer loading, device fallback, tokenization, confidence ranges, edge cases, and batched inferences.
- `tests/test_sentiment_factory.py`: 4 tests validating dynamic model selection, fallback logic, and uniform interfaces.
- `tests/test_sentiment.py`: 6 tests verifying TF-IDF baseline loading and inferences.
- `tests/test_api.py`: 10 tests verifying `/health`, `/model/info`, `/clusters`, `/predict`, `/predict/batch`, and `/insights`.
- `tests/test_clustering.py`: 5 tests verifying K-Means cluster assignment and theme inference.
- `tests/test_pipeline.py`: 4 tests verifying end-to-end processing.
- `tests/test_data_loader.py`: 8 tests verifying CSV/JSON data reading and validation.
- `tests/test_preprocessing.py`: 10 tests verifying text normalization and validators.

---

## 14. P1 → P3 Inference Pipeline & Data Handoff

The system provides a dedicated batch inference and data segregation pipeline designed for seamless inter-phase workflow execution:

```text
Raw comments
      ↓
DistilBERT
      ↓
Sentiment-labeled data
      ↓
┌─────────────┬─────────────┬─────────────┐
│ positive    │ negative    │ neutral     │
│ CSV         │ CSV         │ CSV         │
└─────────────┴─────────────┴─────────────┘
                    ↓
             negative.csv
                    ↓
             P3 clustering
```

### Workflow Responsibilities:
1. **P1 Responsibility (Sentiment Classification & Segregation)**:
   - Receives raw comments CSV (e.g. `data/raw/comments.csv`).
   - Runs the fine-tuned DistilBERT model using batched inference with GPU acceleration (or automatic CPU fallback).
   - Generates predictions including `sentiment` and `confidence` while preserving original comments, IDs, and metadata (`brand`, `platform`).
   - Validates the count invariant:
     $$\text{positive\_count} + \text{negative\_count} + \text{neutral\_count} = \text{total\_processed}$$
   - Automatically segregates results into three distinct output files:
     - `data/processed/positive.csv`
     - `data/processed/negative.csv`
     - `data/processed/neutral.csv`

2. **P3 Responsibility (Complaint Clustering)**:
   - **Consumes**: `data/processed/negative.csv` as its primary input.
   - **Expected Schema**: Each row contains at minimum `id`, `comment`, and `sentiment` (along with `confidence`, `brand`, `platform`).
   - Runs negative comments through the complaint clustering engine (TF-IDF + K-Means) to extract complaint themes, cluster IDs, and actionable insights.

### Running Batch Inference & CSV Segregation:
```powershell
python scripts/label_csv.py `
    --input data/raw/comments.csv `
    --output-dir data/processed/ `
    --batch-size 32
```

---

## 15. Limitations

1. **CPU Latency**: While DistilBERT is optimized (~66M parameters), CPU inference on local Windows hardware averages ~35–60ms per comment compared to ~0.5ms for TF-IDF. For ultra-high throughput without a GPU, the TF-IDF engine can be selected via `SENTIMENT_MODEL=tfidf`.
2. **Context Window**: Comments are tokenized up to a `max_length` of 128 tokens, which covers >99% of social media comments but truncates long blog posts or reviews.
3. **Hard Clustering**: K-Means assigns each negative feedback comment to a single cluster, whereas real complaints occasionally span multiple themes simultaneously.

---

## 16. Future Improvements

1. **Quantization & ONNX Runtime**: Export DistilBERT to ONNX format with INT8 quantization to achieve sub-10ms CPU latency.
2. **Multi-label Complaint Tagging**: Train a multi-label classification head to detect multi-issue complaints (e.g., Delivery Delay AND Damaged Item).
3. **Aspect-Based Sentiment Analysis (ABSA)**: Parse sentiment for distinct entities (e.g., Delivery vs. Product Quality) within a single post.
4. **Active Learning**: Store low-confidence inferences for periodic retraining.

---

## 17. Sentence Embedding Clustering (Member 3)

This section documents the upgraded complaint clustering pipeline that replaces sparse TF-IDF vectors with dense semantic sentence embeddings.

### 17.1 — What Changed

| Component | Before | After |
|-----------|--------|-------|
| Feature representation | TF-IDF (5000-dim sparse) | Sentence Embeddings (384-dim dense) |
| Algorithm | K-Means only | K-Means **and** DBSCAN |
| Semantic understanding | Word frequency | Meaning / context |

### 17.2 — Installation

`sentence-transformers` is already included in `requirements.txt`. Install all dependencies:

```bash
pip install -r requirements.txt
```

The model (`all-MiniLM-L6-v2`) is downloaded automatically on first use from Hugging Face Hub.

### 17.3 — Training the Clustering Model

**K-Means on sentence embeddings** (recommended, deterministic):

```python
from src.clustering.clusterer import ComplaintClusterer

clusterer = ComplaintClusterer()
clusterer.train_embeddings(texts, n_clusters=7)
result = clusterer.predict("My order never arrived")
# → {"cluster_id": 2, "cluster_label": "cluster_2", "complaint_category": "Delivery Problems"}
```

**DBSCAN on sentence embeddings** (density-based, no need to choose K upfront):

```python
clusterer.train_dbscan(texts, eps=0.4, min_samples=5)
# cluster_id = -1 means the comment is a noise point (no cluster)
```

### 17.4 — Switching Between Methods

| Method | Constructor Argument | When to Use |
|--------|---------------------|-------------|
| TF-IDF K-Means (legacy) | `clusterer.train(texts, n_clusters)` | Fast baseline, no GPU needed |
| Embedding K-Means | `clusterer.train_embeddings(texts, n_clusters)` | Best silhouette score, deterministic |
| Embedding DBSCAN | `clusterer.train_dbscan(texts, eps, min_samples)` | Handles outliers, no K required |

### 17.5 — Exporting Clustered Results for the Dashboard

Run the export script to generate `data/processed/clustered_comments.csv`:

```bash
python scripts/export_clustered_complaints.py
```

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `platform` | Social media platform (twitter, instagram, etc.) |
| `brand` | Brand name (Amazon, PhonePe, etc.) |
| `comment` | Original comment text |
| `sentiment` | positive / neutral / negative |
| `cluster_id` | Integer cluster ID (−1 = noise for DBSCAN) |
| `cluster_label` | Short label (e.g., `cluster_2`) |
| `complaint_category` | Business theme (e.g., Delivery Problems) |

**Loading in the dashboard (Member 4):**

```python
import pandas as pd
df = pd.read_csv("data/processed/clustered_comments.csv")
negative_df = df[df["sentiment"] == "negative"]
```

### 17.6 — Evaluating Cluster Quality

Run the K-selection evaluation script to find the best number of clusters:

```bash
python scripts/evaluate_embeddings_clustering.py
```

Compare K-Means vs DBSCAN:

```bash
python scripts/compare_clustering_methods.py
# Output: reports/clustering_comparison.json
```

### 17.7 — Running Tests

**Unit tests (fast, no model download):**

```bash
python -m pytest tests/test_clustering.py -v -m "not integration"
```

**Integration test (downloads model, runs end-to-end):**

```bash
python -m pytest tests/test_clustering.py -v -m "integration"
```

### 17.8 — Known Limitations

- **Cluster labels are generic** (`cluster_0`, `cluster_1`, …) when no TF-IDF vectorizer is used. The `infer_theme_category` function maps them to business themes using keyword heuristics.
- **DBSCAN noise points** (`cluster_id = -1`) are valid outputs — they represent comments that don't fit into any dense cluster.
- **Silhouette scores across different feature spaces** (TF-IDF vs embeddings) are not directly comparable.
- **Model download required on first run** — the `all-MiniLM-L6-v2` model (~90 MB) is fetched from Hugging Face Hub automatically.

