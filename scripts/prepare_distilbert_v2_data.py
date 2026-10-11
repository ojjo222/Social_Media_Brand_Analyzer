import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.distilbert_analyzer import clean_text_for_transformer


def calculate_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_datasets(
    raw_batches_dir: str = "data/raw",
    benchmark_dir: str = "data/benchmark",
    output_dir: str = "data/benchmark/distilbert_v2"
) -> Dict[str, Any]:
    raw_path = Path(raw_batches_dir)
    bench_path = Path(benchmark_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    orig_test_path = bench_path / "test.csv"
    orig_val_path = bench_path / "validation.csv"
    orig_train_path = bench_path / "train.csv"

    if not orig_test_path.exists():
        raise FileNotFoundError(f"Original test benchmark file missing: {orig_test_path}")
    if not orig_val_path.exists():
        raise FileNotFoundError(f"Original validation benchmark file missing: {orig_val_path}")

    # 1. Load original benchmark validation and test splits
    test_df = pd.read_csv(orig_test_path)
    val_df = pd.read_csv(orig_val_path)
    orig_train_df = pd.read_csv(orig_train_path) if orig_train_path.exists() else None

    # Calculate normalized text representations (lowercased for uncased transformer leakage prevention)
    normalize_fn = lambda t: clean_text_for_transformer(t).lower()
    test_df["clean_text"] = test_df["text"].apply(normalize_fn)
    val_df["clean_text"] = val_df["text"].apply(normalize_fn)

    test_clean_set = set(test_df["clean_text"])
    val_clean_set = set(val_df["clean_text"])

    # Strict assertion: original val and test must not overlap
    val_test_overlap = val_clean_set.intersection(test_clean_set)
    if val_test_overlap:
        raise ValueError(f"Data integrity error: {len(val_test_overlap)} overlapping texts between original val and test.")

    # 2. Locate and load all 16 batches
    batch_pattern = str(raw_path / "comments_batch_*.csv")
    batch_files = sorted(glob.glob(batch_pattern))
    if len(batch_files) < 16:
        raise FileNotFoundError(f"Expected 16 batch CSVs in {raw_path}, found {len(batch_files)}.")

    batch_dfs = []
    batch_info = []
    for bf in batch_files:
        df_b = pd.read_csv(bf)
        batch_dfs.append(df_b)
        batch_info.append({
            "file": os.path.basename(bf),
            "rows": len(df_b),
            "columns": list(df_b.columns)
        })

    all_raw_df = pd.concat(batch_dfs, ignore_index=True)
    total_raw_rows = len(all_raw_df)

    # 3. Clean and normalize raw candidates
    all_raw_df["clean_text"] = all_raw_df["text"].apply(normalize_fn)
    # Remove any empty or non-string normalized text
    non_empty_mask = all_raw_df["clean_text"].str.strip().str.len() > 0
    clean_pool = all_raw_df[non_empty_mask].copy()
    empty_excluded = total_raw_rows - len(clean_pool)

    # 4. Leakage prevention: exclude any candidate present in test or val
    test_overlap_mask = clean_pool["clean_text"].isin(test_clean_set)
    val_overlap_mask = clean_pool["clean_text"].isin(val_clean_set)

    test_excluded_count = int(test_overlap_mask.sum())
    val_excluded_count = int(val_overlap_mask.sum())

    pool_no_leakage = clean_pool[~test_overlap_mask & ~val_overlap_mask].copy()

    # 5. Conflicting label resolution
    # Detect identical clean_text with divergent sentiment labels
    label_group = pool_no_leakage.groupby("clean_text")["sentiment"].unique()
    conflicting_texts = set(label_group[label_group.apply(len) > 1].index)
    conflict_mask = pool_no_leakage["clean_text"].isin(conflicting_texts)
    conflict_excluded_count = int(conflict_mask.sum())
    conflicts_detail = {
        k: list(v) for k, v in label_group[label_group.apply(len) > 1].items()
    }

    pool_consistent = pool_no_leakage[~conflict_mask].copy()

    # 6. Deduplication on clean_text
    dup_mask = pool_consistent.duplicated(subset=["clean_text"], keep="first")
    duplicate_excluded_count = int(dup_mask.sum())

    train_df = pool_consistent[~dup_mask].copy()

    # Drop temporary clean_text column so schemas strictly match original
    cols_to_save = ["text", "sentiment", "brand", "platform", "source"]
    train_final = train_df[cols_to_save].copy()
    val_final = val_df[cols_to_save].copy()

    # 7. Write final train, validation, and test files
    out_train_path = out_path / "train.csv"
    out_val_path = out_path / "validation.csv"
    out_test_path = out_path / "test.csv"

    train_final.to_csv(out_train_path, index=False, encoding="utf-8")
    val_final.to_csv(out_val_path, index=False, encoding="utf-8")

    # For test.csv: copy original byte-for-byte to prevent any encoding/formatting differences
    with open(orig_test_path, "rb") as src, open(out_test_path, "wb") as dst:
        dst.write(src.read())

    # 8. Byte-for-byte verification of test.csv
    orig_test_md5 = calculate_md5(orig_test_path)
    new_test_md5 = calculate_md5(out_test_path)
    if orig_test_md5 != new_test_md5:
        raise ValueError("Byte-for-byte verification failed: test.csv differs from original benchmark test.csv")

    # 9. Leakage assertion across all 3 finalized splits
    final_train_clean = set(train_final["text"].apply(normalize_fn))
    final_val_clean = set(val_final["text"].apply(normalize_fn))
    final_test_clean = set(pd.read_csv(out_test_path)["text"].apply(normalize_fn))

    assert len(final_train_clean.intersection(final_test_clean)) == 0, "FATAL: Leakage between Train and Test!"
    assert len(final_train_clean.intersection(final_val_clean)) == 0, "FATAL: Leakage between Train and Validation!"
    assert len(final_val_clean.intersection(final_test_clean)) == 0, "FATAL: Leakage between Validation and Test!"

    summary = {
        "version": "distilbert_v2",
        "raw_batches_inspected": len(batch_files),
        "total_raw_rows_in_batches": total_raw_rows,
        "batch_details": batch_info,
        "exclusions": {
            "empty_or_whitespace_text": empty_excluded,
            "test_set_leakage_matches": test_excluded_count,
            "validation_set_leakage_matches": val_excluded_count,
            "conflicting_sentiment_labels": conflict_excluded_count,
            "conflicting_groups_count": len(conflicting_texts),
            "conflicting_groups_sample": conflicts_detail,
            "normalized_duplicates_within_train": duplicate_excluded_count
        },
        "splits": {
            "train": {
                "rows": len(train_final),
                "sentiment_distribution": train_final["sentiment"].value_counts().to_dict(),
                "source_distribution": train_final["source"].value_counts().to_dict(),
                "platform_distribution": train_final["platform"].value_counts().to_dict(),
                "md5": calculate_md5(out_train_path)
            },
            "validation": {
                "rows": len(val_final),
                "sentiment_distribution": val_final["sentiment"].value_counts().to_dict(),
                "source_distribution": val_final["source"].value_counts().to_dict(),
                "platform_distribution": val_final["platform"].value_counts().to_dict(),
                "md5": calculate_md5(out_val_path)
            },
            "test": {
                "rows": len(test_df),
                "sentiment_distribution": test_df["sentiment"].value_counts().to_dict(),
                "source_distribution": test_df["source"].value_counts().to_dict(),
                "platform_distribution": test_df["platform"].value_counts().to_dict(),
                "md5": new_test_md5,
                "byte_for_byte_identical_to_original": True
            }
        },
        "leakage_verification": {
            "train_test_overlap": 0,
            "train_val_overlap": 0,
            "val_test_overlap": 0,
            "status": "PASSED"
        }
    }

    summary_file = out_path / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("Preparing leakage-safe DistilBERT v2 benchmark dataset...")
    summary = prepare_datasets()
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Raw Batch Rows Inspected: {summary['total_raw_rows_in_batches']:,}")
    print("\nExclusions:")
    for k, v in summary["exclusions"].items():
        if k != "conflicting_groups_sample":
            print(f"  - {k}: {v}")
    print("\nFinal Prepared Splits:")
    print(f"  - Train:      {summary['splits']['train']['rows']:,} rows")
    print(f"  - Validation: {summary['splits']['validation']['rows']:,} rows")
    print(f"  - Test:       {summary['splits']['test']['rows']:,} rows (Byte-for-byte match: {summary['splits']['test']['byte_for_byte_identical_to_original']})")
    print("\nLeakage Check:")
    print(f"  - Train / Test overlap:       {summary['leakage_verification']['train_test_overlap']}")
    print(f"  - Train / Validation overlap: {summary['leakage_verification']['train_val_overlap']}")
    print(f"  - Val / Test overlap:         {summary['leakage_verification']['val_test_overlap']}")
    print(f"  - Status:                     {summary['leakage_verification']['status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
