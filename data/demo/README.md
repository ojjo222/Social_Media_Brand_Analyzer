# Dashboard Demo & Fixture Dataset (100,000 Records)

## Overview
This directory contains a pre-generated testing fixture dataset specifically designed for the **Dashboard Teammate (P3)** to test and optimize the Streamlit dashboard, charts, filters, sentiment distributions, and cluster visualizations at high scale (~100,000 comments) without waiting for full transformer inference.

## File Location
- File: `data/demo/dashboard_comments_100k.csv`
- Records: 100,000
- Unique IDs: 100,000

## Schema Specification
The fixture strictly matches the schema consumed by `src/dashboard/app.py`:

| Column Name | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier (`DEMO_000001` to `DEMO_100000`) |
| `comment` | string | Full text of customer comment/review |
| `original_text` | string | Preserved text for cluster drill-down display in Tab 2 |
| `sentiment` | string | Classification: `'positive'`, `'negative'`, or `'neutral'` |
| `confidence` | float | Calibrated model confidence score in `[0.70, 0.99]` |
| `sentiment_confidence` | float | Alias field for direct compatibility with API schemas |
| `is_negative` | boolean | `True` for negative sentiment, `False` otherwise |
| `brand` | string | Associated brand or entity (Amazon, Twitter, Google Play apps, etc.) |
| `platform` | string | Origin platform (`amazon`, `twitter`, `google_play`, `youtube`, `reddit`, `yelp`) |
| `source` | string | Underlying dataset source identifier |
| `complaint_cluster` | int / null | Assigned cluster ID (0 to 9) for negative comments; null for others |
| `cluster_id` | int / null | Duplicate alias field for cluster ID compatibility |
| `complaint_category` | string / null | Operational complaint theme (e.g. `'Customer Support'`, `'Delivery Problems'`) |
| `provenance` | string | Constant identifier: `'demo_fixture_for_dashboard_development'` |

## Distribution Summary
- **Total Comments**: 100,000
- **Sentiments**:
  - Positive: 49,117 (49.1%)
  - Neutral: 18,854 (18.9%)
  - Negative: 32,029 (32.0%)
- **Platforms**:
  - amazon: 26,667
  - twitter: 20,000
  - google_play: 16,667
  - youtube: 16,667
  - reddit: 13,333
  - yelp: 6,666
- **Complaint Themes (Negative Feedback Only)**:
  - App & Technical Bugs: 7,997
  - Delivery Problems: 7,156
  - Customer Support: 7,040
  - Refund Issues: 5,083
  - Product Quality: 4,753

## How to Use in the Dashboard
The dashboard teammate can directly load this fixture in `src/dashboard/app.py` or via Streamlit's sidebar:
```python
# Direct load in dashboard session state:
import pandas as pd
demo_df = pd.read_csv("data/demo/dashboard_comments_100k.csv")
st.session_state.batch_results = demo_df
```
Or upload `data/demo/dashboard_comments_100k.csv` directly using the Streamlit CSV uploader in the sidebar.

## Provenance & Attribution Warning
> [!IMPORTANT]
> This dataset is a **synthetic testing fixture** created exclusively for dashboard UI/UX development and performance testing.
> Sentiment labels and complaint categories are simulated demo annotations.
> **DO NOT** use this fixture for model training, validation, or test evaluation.
> **DO NOT** represent these predictions as genuine DistilBERT v2 production outputs.
