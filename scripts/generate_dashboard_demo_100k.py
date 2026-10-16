"""
Generate 100,000 demo/fixture comment records for dashboard testing.

This dataset provides the dashboard developer with a realistic, high-volume
fixture (~100,000 records) to test charts, filters, metrics, and cluster summaries
without waiting for full production transformer inference.

IMPORTANT:
These records are explicitly tagged with `provenance = 'demo_fixture_for_dashboard_development'`.
They are NOT actual DistilBERT v2 production predictions.
"""

import os
import sys
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_RAW_150K = PROJECT_ROOT / "data" / "processed" / "inference_150k" / "raw_comments_150k.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "demo"
OUTPUT_FILE = OUTPUT_DIR / "dashboard_comments_100k.csv"
README_FILE = OUTPUT_DIR / "README.md"

THEME_KEYWORDS = {
    "Delivery Problems": (["delivery", "late", "tracking", "delivered", "order", "package", "delay", "wait"], [0, 1]),
    "Customer Support": (["customer", "support", "service", "hold", "bot", "human", "agent", "reply", "call", "rude"], [4]),
    "Refund Issues": (["refund", "money", "bank", "reflecting", "return", "cancel", "deducted", "transaction"], [2, 5]),
    "App & Technical Bugs": (["app", "crash", "bug", "buggy", "login", "checkout", "error", "website", "pay", "update", "freeze"], [3, 6, 7]),
    "Product Quality": (["quality", "broken", "cold", "stale", "packaging", "item", "food", "damaged", "fake", "bad", "cheap"], [8, 9])
}

POS_WORDS = {"great", "love", "awesome", "best", "excellent", "good", "perfect", "amazing", "fast", "thanks", "helpful", "recommend", "smooth", "happy", "nice", "fantastic", "wonderful"}
NEG_WORDS = {"terrible", "worst", "broken", "delay", "crash", "bug", "scam", "waste", "horrible", "refund", "poor", "hate", "issue", "problem", "late", "damage", "sucks", "disappointed", "never", "awful", "useless", "garbage"}


def generate_demo_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    if not INPUT_RAW_150K.exists():
        raise FileNotFoundError(f"Input corpus not found at {INPUT_RAW_150K}. Run scripts/build_150k_multi_source_dataset.py first.")

    print(f"Loading raw comments from {INPUT_RAW_150K}...")
    df_raw = pd.read_csv(INPUT_RAW_150K)

    # Stratified selection to reach exactly 100,000 comments across all 6 platforms
    target_100k = 100000
    stratified_counts = {
        "amazon": 26667,
        "twitter": 20000,
        "google_play": 16667,
        "youtube": 16667,
        "reddit": 13333,
        "yelp": 6666
    }
    # Adjust total to exactly 100,000
    total_strat = sum(stratified_counts.values())
    stratified_counts["amazon"] += (target_100k - total_strat)

    selected_dfs = []
    for plat, count in stratified_counts.items():
        plat_df = df_raw[df_raw["platform"] == plat]
        if len(plat_df) >= count:
            selected_dfs.append(plat_df.sample(n=count, random_state=42))
        else:
            selected_dfs.append(plat_df)

    df_100k = pd.concat(selected_dfs, ignore_index=True)
    if len(df_100k) > target_100k:
        df_100k = df_100k.iloc[:target_100k]
    elif len(df_100k) < target_100k:
        needed = target_100k - len(df_100k)
        remaining = df_raw[~df_raw["id"].isin(df_100k["id"])].sample(n=needed, random_state=42)
        df_100k = pd.concat([df_100k, remaining], ignore_index=True)

    print(f"Selected {len(df_100k)} records across {df_100k['platform'].nunique()} platforms.")

    records = []
    category_list = list(THEME_KEYWORDS.keys())

    for idx, row in df_100k.iterrows():
        comment_text = str(row["comment"])
        lower_text = comment_text.lower()
        words = set(lower_text.split())

        # Determine sentiment via lexical cues + realistic distribution
        pos_score = len(words.intersection(POS_WORDS))
        neg_score = len(words.intersection(NEG_WORDS))

        rand_val = random.random()
        if neg_score > pos_score or rand_val < 0.25:
            sentiment = "negative"
            confidence = round(random.uniform(0.85, 0.99), 4)
            is_negative = True

            # Match theme
            matched_theme = None
            for theme, (kws, cluster_choices) in THEME_KEYWORDS.items():
                if any(kw in lower_text for kw in kws):
                    matched_theme = theme
                    cluster_id = random.choice(cluster_choices)
                    break

            if not matched_theme:
                matched_theme = random.choice(category_list)
                cluster_id = random.choice(THEME_KEYWORDS[matched_theme][1])

            complaint_category = matched_theme
            complaint_cluster = cluster_id
        elif pos_score > neg_score or rand_val < 0.70:
            sentiment = "positive"
            confidence = round(random.uniform(0.82, 0.99), 4)
            is_negative = False
            complaint_category = None
            complaint_cluster = None
        else:
            sentiment = "neutral"
            confidence = round(random.uniform(0.70, 0.94), 4)
            is_negative = False
            complaint_category = None
            complaint_cluster = None

        demo_id = f"DEMO_{idx + 1:06d}"
        records.append({
            "id": demo_id,
            "comment": comment_text,
            "original_text": comment_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "sentiment_confidence": confidence,
            "is_negative": is_negative,
            "brand": row.get("brand", "Unknown"),
            "platform": row.get("platform", "Unknown"),
            "source": row.get("source", "raw_corpus"),
            "complaint_cluster": complaint_cluster,
            "cluster_id": complaint_cluster,
            "complaint_category": complaint_category,
            "provenance": "demo_fixture_for_dashboard_development"
        })

    demo_df = pd.DataFrame(records)

    # Save CSV
    demo_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(demo_df)} demo records to {OUTPUT_FILE}")

    # Generate companion README.md
    readme_content = f"""# Dashboard Demo & Fixture Dataset (100,000 Records)

## Overview
This directory contains a pre-generated testing fixture dataset specifically designed for the **Dashboard Teammate (P3)** to test and optimize the Streamlit dashboard, charts, filters, sentiment distributions, and cluster visualizations at high scale (~100,000 comments) without waiting for full transformer inference.

## File Location
- File: `data/demo/dashboard_comments_100k.csv`
- Records: {len(demo_df):,}
- Unique IDs: {demo_df['id'].nunique():,}

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
- **Total Comments**: {len(demo_df):,}
- **Sentiments**:
  - Positive: {int((demo_df['sentiment'] == 'positive').sum()):,} ({((demo_df['sentiment'] == 'positive').mean()*100):.1f}%)
  - Neutral: {int((demo_df['sentiment'] == 'neutral').sum()):,} ({((demo_df['sentiment'] == 'neutral').mean()*100):.1f}%)
  - Negative: {int((demo_df['sentiment'] == 'negative').sum()):,} ({((demo_df['sentiment'] == 'negative').mean()*100):.1f}%)
- **Platforms**:
{chr(10).join([f"  - {k}: {v:,}" for k, v in demo_df['platform'].value_counts().items()])}
- **Complaint Themes (Negative Feedback Only)**:
{chr(10).join([f"  - {k}: {v:,}" for k, v in demo_df['complaint_category'].dropna().value_counts().items()])}

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
"""

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved companion README to {README_FILE}")


if __name__ == "__main__":
    generate_demo_dataset()
