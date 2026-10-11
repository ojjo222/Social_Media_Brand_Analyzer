import json
import hashlib
from pathlib import Path
import pandas as pd
import pytest

from src.sentiment.distilbert_analyzer import clean_text_for_transformer


BENCHMARK_V2_DIR = Path("data/benchmark/distilbert_v2")
ORIG_BENCHMARK_DIR = Path("data/benchmark")

pytestmark = pytest.mark.skipif(
    not (BENCHMARK_V2_DIR / "train.csv").exists() or not (ORIG_BENCHMARK_DIR / "test.csv").exists(),
    reason="distilbert_v2 benchmark dataset not present locally"
)


def calculate_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_v2_files_exist():
    assert (BENCHMARK_V2_DIR / "train.csv").exists()
    assert (BENCHMARK_V2_DIR / "validation.csv").exists()
    assert (BENCHMARK_V2_DIR / "test.csv").exists()
    assert (BENCHMARK_V2_DIR / "dataset_summary.json").exists()


def test_byte_for_byte_test_preservation():
    orig_test = ORIG_BENCHMARK_DIR / "test.csv"
    v2_test = BENCHMARK_V2_DIR / "test.csv"

    assert orig_test.stat().st_size == v2_test.stat().st_size
    assert calculate_md5(orig_test) == calculate_md5(v2_test)


def test_validation_set_preservation():
    orig_val = ORIG_BENCHMARK_DIR / "validation.csv"
    v2_val = BENCHMARK_V2_DIR / "validation.csv"

    df_orig = pd.read_csv(orig_val)
    df_v2 = pd.read_csv(v2_val)

    assert len(df_v2) == len(df_orig) == 7473
    assert list(df_v2.columns) == list(df_orig.columns)


def test_zero_leakage_between_all_splits():
    train_df = pd.read_csv(BENCHMARK_V2_DIR / "train.csv")
    val_df = pd.read_csv(BENCHMARK_V2_DIR / "validation.csv")
    test_df = pd.read_csv(BENCHMARK_V2_DIR / "test.csv")

    normalize_fn = lambda t: clean_text_for_transformer(t).lower()

    train_clean = set(train_df["text"].apply(normalize_fn))
    val_clean = set(val_df["text"].apply(normalize_fn))
    test_clean = set(test_df["text"].apply(normalize_fn))

    assert len(train_clean.intersection(test_clean)) == 0, "Train and Test share normalized text!"
    assert len(train_clean.intersection(val_clean)) == 0, "Train and Validation share normalized text!"
    assert len(val_clean.intersection(test_clean)) == 0, "Validation and Test share normalized text!"


def test_train_metadata_columns_and_no_missing():
    train_df = pd.read_csv(BENCHMARK_V2_DIR / "train.csv")
    expected_cols = ["text", "sentiment", "brand", "platform", "source"]

    assert list(train_df.columns) == expected_cols
    assert train_df.isnull().sum().sum() == 0
    assert len(train_df) > 100000


def test_no_conflicting_sentiment_labels_in_train():
    train_df = pd.read_csv(BENCHMARK_V2_DIR / "train.csv")
    normalize_fn = lambda t: clean_text_for_transformer(t).lower()
    train_df["clean_text"] = train_df["text"].apply(normalize_fn)

    label_groups = train_df.groupby("clean_text")["sentiment"].nunique()
    conflicts = label_groups[label_groups > 1]
    assert len(conflicts) == 0, f"Found {len(conflicts)} conflicting text groups in train.csv!"


def test_sentiment_labels_valid_distribution():
    train_df = pd.read_csv(BENCHMARK_V2_DIR / "train.csv")
    valid_labels = {"positive", "negative", "neutral"}

    assert set(train_df["sentiment"].unique()).issubset(valid_labels)
    counts = train_df["sentiment"].value_counts()
    assert counts["positive"] > 0
    assert counts["negative"] > 0
    assert counts["neutral"] > 0


def test_dataset_summary_json_integrity():
    summary_path = BENCHMARK_V2_DIR / "dataset_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["version"] == "distilbert_v2"
    assert summary["raw_batches_inspected"] == 16
    assert summary["leakage_verification"]["status"] == "PASSED"
    assert summary["leakage_verification"]["train_test_overlap"] == 0
    assert summary["leakage_verification"]["train_val_overlap"] == 0
    assert summary["leakage_verification"]["val_test_overlap"] == 0
    assert summary["splits"]["test"]["byte_for_byte_identical_to_original"] is True
