import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.factory import get_sentiment_analyzer
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("label_csv")

CANDIDATE_COMMENT_COLS = ["comment", "text", "content", "body", "review", "comment_text"]
CANDIDATE_ID_COLS = ["id", "comment_id", "ID", "Id"]


def detect_comment_column(df: pd.DataFrame, specified_col: Optional[str] = None) -> str:
    if specified_col:
        if specified_col in df.columns:
            return specified_col
        raise ValueError(
            f"Specified comment column '{specified_col}' not found. Available columns: {list(df.columns)}"
        )

    for col in CANDIDATE_COMMENT_COLS:
        if col in df.columns:
            return col

    # Check case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for col in CANDIDATE_COMMENT_COLS:
        if col in lower_map:
            return lower_map[col]

    raise ValueError(
        f"Could not automatically detect a comment column. Available columns: {list(df.columns)}. "
        f"Please specify using --comment-col."
    )


def detect_id_column(df: pd.DataFrame, specified_col: Optional[str] = None) -> Optional[str]:
    if specified_col:
        if specified_col in df.columns:
            return specified_col
        raise ValueError(
            f"Specified ID column '{specified_col}' not found. Available columns: {list(df.columns)}"
        )

    for col in CANDIDATE_ID_COLS:
        if col in df.columns:
            return col
    return None


def process_and_segregate_csv(
    input_file: str,
    output_dir: str = "data/processed",
    batch_size: int = 32,
    comment_col: Optional[str] = None,
    id_col: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading input CSV from %s", input_path)
    df = pd.read_csv(input_path)

    raw_count = len(df)
    if raw_count == 0:
        logger.warning("Input CSV is empty (0 rows). Creating empty output files.")
        empty_cols = ["id", "comment", "sentiment", "confidence"]
        df_empty = pd.DataFrame(columns=empty_cols)
        for name in ["positive.csv", "negative.csv", "neutral.csv"]:
            df_empty.to_csv(out_dir / name, index=False)
        stats = {
            "total_processed": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0
        }
        return df_empty, df_empty, df_empty, stats

    # 1. Detect comment column
    source_comment_col = detect_comment_column(df, comment_col)
    logger.info("Using '%s' as the comment column", source_comment_col)

    # 2. Preserve original comment and handle missing
    original_comments = df[source_comment_col].fillna("").astype(str)
    df["comment"] = original_comments

    # 3. Handle ID column
    detected_id = detect_id_column(df, id_col)
    if detected_id:
        logger.info("Preserving existing ID column '%s'", detected_id)
        if detected_id != "id":
            df["id"] = df[detected_id]
        # Check for duplicate IDs
        num_duplicates = df["id"].duplicated().sum()
        if num_duplicates > 0:
            logger.warning("Found %d duplicate IDs in input CSV.", num_duplicates)
    else:
        logger.info("No ID column found. Generating 1-indexed sequential IDs.")
        df["id"] = range(1, raw_count + 1)

    # 4. Load DistilBERT model
    logger.info("Initializing sentiment analyzer (model_dir=%s, device=%s)", model_dir, device)
    analyzer = get_sentiment_analyzer(model_name="distilbert", model_dir=model_dir, device=device)
    if not analyzer.is_loaded():
        raise RuntimeError("Failed to load sentiment model.")

    # 5. Batched inference
    texts = original_comments.tolist()
    sentiments = []
    confidences = []

    logger.info("Running batched sentiment inference on %d comments (batch_size=%d)...", len(texts), batch_size)
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring comments"):
        batch_texts = texts[i : i + batch_size]
        batch_preds = analyzer.predict_batch(batch_texts, batch_size=batch_size)
        for pred in batch_preds:
            sentiments.append(pred["sentiment"])
            confidences.append(round(float(pred["confidence"]), 4))

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    # 6. Column ordering: id, comment, sentiment, confidence, followed by metadata
    leading_cols = ["id", "comment", "sentiment", "confidence"]
    remaining_cols = [c for c in df.columns if c not in leading_cols and c != source_comment_col]
    ordered_cols = leading_cols + remaining_cols
    df_ordered = df[ordered_cols]

    # 7. Segregate predictions
    df_pos = df_ordered[df_ordered["sentiment"] == "positive"].copy()
    df_neg = df_ordered[df_ordered["sentiment"] == "negative"].copy()
    df_neu = df_ordered[df_ordered["sentiment"] == "neutral"].copy()

    pos_count = len(df_pos)
    neg_count = len(df_neg)
    neu_count = len(df_neu)

    # 8. Data validation invariant
    if pos_count + neg_count + neu_count != raw_count:
        raise ValueError(
            f"Data validation failed: positive ({pos_count}) + negative ({neg_count}) + "
            f"neutral ({neu_count}) != total rows ({raw_count})"
        )

    # 9. Save segregated CSVs
    pos_file = out_dir / "positive.csv"
    neg_file = out_dir / "negative.csv"
    neu_file = out_dir / "neutral.csv"

    df_pos.to_csv(pos_file, index=False)
    df_neg.to_csv(neg_file, index=False)
    df_neu.to_csv(neu_file, index=False)

    stats = {
        "total_processed": raw_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_file": str(pos_file),
        "negative_file": str(neg_file),
        "neutral_file": str(neu_file)
    }

    return df_pos, df_neg, df_neu, stats


def print_summary(stats: Dict[str, Any]):
    print("\n" + "=" * 50)
    print(f"Processed: {stats['total_processed']} comments\n")
    print(f"Positive: {stats['positive_count']}")
    print(f"Negative: {stats['negative_count']}")
    print(f"Neutral:  {stats['neutral_count']}\n")
    print(f"Total:    {stats['total_processed']}\n")
    print("Output:")
    print(f"  {stats.get('positive_file', 'positive.csv')}")
    print(f"  {stats.get('negative_file', 'negative.csv')}")
    print(f"  {stats.get('neutral_file', 'neutral.csv')}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run DistilBERT sentiment analysis on raw comments CSV and segregate by sentiment."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input raw comments CSV file (e.g. data/raw/comments.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed",
        help="Directory to save positive.csv, negative.csv, neutral.csv (default: data/processed)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)"
    )
    parser.add_argument(
        "--comment-col",
        default=None,
        help="Explicit name of the text/comment column (auto-detected if omitted)"
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Explicit name of the ID column (auto-detected or generated if omitted)"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to DistilBERT model directory (default: models/distilbert)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device ('cpu', 'cuda', etc., auto-detected if omitted)"
    )

    args = parser.parse_args()

    try:
        _, _, _, stats = process_and_segregate_csv(
            input_file=args.input,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            comment_col=args.comment_col,
            id_col=args.id_col,
            model_dir=args.model_dir,
            device=args.device
        )
        print_summary(stats)
    except Exception as e:
        logger.error("Processing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
