# Member 1 — Data Collection & Preprocessing

This branch contains the data pipeline for the Social Media Brand Analyzer project.

## What I built

- Collected and cleaned 75,542 real comments from Kaggle (YouTube + Twitter)
- Generated 75,000+ synthetic comments using 3 LLM APIs (Groq, Gemini, OpenRouter)
- Combined everything into one dataset with columns: `text`, `sentiment`, `brand`, `platform`, `source`
- Filtered to English-only using `langdetect` (removed ~5% non-English)
- Split into 10k-row batches for delivery to Member 2

## Files

All scripts and data are in `data_collection/`.

### Scripts

| File | Purpose |
|------|---------|
| `clean_kaggle.py` | Cleans the Kaggle YouTube dataset — drops nulls, duplicates, standardizes sentiment labels |
| `clean_twitter.py` | Cleans the Kaggle Twitter dataset — drops nulls, duplicates, drops irrelevant labels |
| `inspect_kaggle.py` | Quick inspection script for the raw Kaggle CSV |
| `generate_groq.py` | Generates synthetic comments using Groq API |
| `generate_gemini.py` | Generates synthetic comments using Google Gemini API |
| `generate_openrouter.py` | Generates synthetic comments using OpenRouter API |
| `combine_all.py` | Merges all sources, deduplicates, splits into 10k-row batches |
| `filter_english.py` | Filters to English-only using `langdetect` |
| `split_english.py` | Splits English dataset into delivery batches |
| `config.py` | Config: brands, sentiment weights, platform targets, API shares |

### Data files

| File | Rows | Description |
|------|------|-------------|
| `kaggle_clean.csv` | 17,871 | Cleaned YouTube comments (Kaggle) |
| `twitter_clean.csv` | 57,671 | Cleaned Twitter comments (Kaggle) |
| `synthetic_groq.csv` | ~75,000 | Synthetic via Groq |
| `synthetic_gemini.csv` | ~1,000 | Synthetic via Gemini |
| `synthetic_openrouter.csv` | ~4,000 | Synthetic via OpenRouter |
| `final_dataset.csv` | 155,008 | Combined (all languages) |
| `final_dataset_english.csv` | 150,115 | **Final deliverable** — English only |

## Pipeline overview

1. **Real data:** Kaggle (YouTube Comments + Twitter Entity Sentiment)
2. **Synthetic data:** Generated via 3 LLM APIs to fill gaps (especially Instagram)
3. **Combining:** Merged, deduplicated, standardized into 5 columns
4. **Language filtering:** English-only via `langdetect`
5. **Delivery:** Split into 10k-row batches, delivered to Member 2

## Technologies used

- **Python** — primary language
- **pandas** — data manipulation
- **langdetect** — language detection
- **emoji** — emoji removal
- **python-dotenv** — env management
- **groq**, **google-genai**, **openai** — API clients
- **Kaggle** — real data source
- **Git + GitHub** — version control

## Final dataset

- **Total rows:** 150,115 (English only)
- **Platforms:** YouTube, Twitter, Instagram
- **Sentiments:** Positive (~50k), Negative (~40k), Neutral (~35k)
- **Brands:** 16 total (Samsung, Apple, OnePlus, Xiaomi, Netflix, Flipkart, Zomato, Swiggy, Amazon India, etc.)
- **Source mix:** ~50% real (Kaggle), ~50% synthetic (LLM-generated)

## How to reproduce

```bash
pip install pandas python-dotenv emoji langdetect google-genai groq openai

python combine_all.py
python filter_english.py
python split_english.py