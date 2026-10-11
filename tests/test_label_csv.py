import pytest
import pandas as pd
from pathlib import Path
from scripts.label_csv import process_and_segregate_csv, detect_comment_column, detect_id_column


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "sample_comments.csv"
    data = {
        "id": [101, 102, 103, 104],
        "comment": [
            "I love this product, absolutely outstanding quality!",
            "Terrible experience, damaged item and late delivery.",
            "What time is your store open today?",
            "Awful customer support, never buying again."
        ],
        "brand": ["BrandA", "BrandB", "BrandC", "BrandB"],
        "platform": ["twitter", "instagram", "facebook", "twitter"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_1_valid_csv_input(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(sample_csv),
        output_dir=str(out_dir),
        batch_size=2
    )

    assert stats["total_processed"] == 4
    assert (out_dir / "positive.csv").exists()
    assert (out_dir / "negative.csv").exists()
    assert (out_dir / "neutral.csv").exists()


def test_2_missing_comment_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    df = pd.DataFrame({"id": [1, 2], "score": [5, 1], "author": ["alice", "bob"]})
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Could not automatically detect a comment column"):
        process_and_segregate_csv(input_file=str(bad_csv), output_dir=str(tmp_path / "out"))


def test_3_missing_id_handling(tmp_path):
    no_id_csv = tmp_path / "no_id.csv"
    df = pd.DataFrame({
        "comment": ["Great job!", "Worst item ever."],
        "brand": ["Alpha", "Beta"]
    })
    df.to_csv(no_id_csv, index=False)

    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(no_id_csv),
        output_dir=str(tmp_path / "out")
    )

    combined = pd.concat([df_pos, df_neg, df_neu], ignore_index=True)
    assert "id" in combined.columns
    assert sorted(combined["id"].tolist()) == [1, 2]


def test_4_empty_and_nan_comments(tmp_path):
    mixed_csv = tmp_path / "mixed.csv"
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": ["", None, "Super fast shipping!"]
    })
    df.to_csv(mixed_csv, index=False)

    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(mixed_csv),
        output_dir=str(tmp_path / "out")
    )

    assert stats["total_processed"] == 3
    # The empty/null comments should safely default to neutral
    assert len(df_neu) >= 2


def test_5_sentiment_segregation(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(sample_csv),
        output_dir=str(out_dir)
    )

    pos_read = pd.read_csv(out_dir / "positive.csv")
    neg_read = pd.read_csv(out_dir / "negative.csv")
    neu_read = pd.read_csv(out_dir / "neutral.csv")

    assert (pos_read["sentiment"] == "positive").all()
    assert (neg_read["sentiment"] == "negative").all()
    assert (neu_read["sentiment"] == "neutral").all()


def test_6_output_row_counts(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(sample_csv),
        output_dir=str(out_dir)
    )

    pos_read = pd.read_csv(out_dir / "positive.csv")
    neg_read = pd.read_csv(out_dir / "negative.csv")
    neu_read = pd.read_csv(out_dir / "neutral.csv")

    assert len(pos_read) == stats["positive_count"]
    assert len(neg_read) == stats["negative_count"]
    assert len(neu_read) == stats["neutral_count"]


def test_7_output_schema_uniformity(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    process_and_segregate_csv(input_file=str(sample_csv), output_dir=str(out_dir))

    pos_read = pd.read_csv(out_dir / "positive.csv")
    neg_read = pd.read_csv(out_dir / "negative.csv")
    neu_read = pd.read_csv(out_dir / "neutral.csv")

    expected_cols = ["id", "comment", "sentiment", "confidence", "brand", "platform"]
    assert list(pos_read.columns) == expected_cols
    assert list(neg_read.columns) == expected_cols
    assert list(neu_read.columns) == expected_cols


def test_8_preservation_of_ids(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    process_and_segregate_csv(input_file=str(sample_csv), output_dir=str(out_dir))

    combined = pd.concat([
        pd.read_csv(out_dir / "positive.csv"),
        pd.read_csv(out_dir / "negative.csv"),
        pd.read_csv(out_dir / "neutral.csv")
    ])

    assert sorted(combined["id"].tolist()) == [101, 102, 103, 104]


def test_9_preservation_of_original_comments(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    process_and_segregate_csv(input_file=str(sample_csv), output_dir=str(out_dir))

    original_df = pd.read_csv(sample_csv)
    combined = pd.concat([
        pd.read_csv(out_dir / "positive.csv"),
        pd.read_csv(out_dir / "negative.csv"),
        pd.read_csv(out_dir / "neutral.csv")
    ]).sort_values("id")

    assert combined["comment"].tolist() == original_df["comment"].tolist()


def test_10_confidence_values_valid(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    process_and_segregate_csv(input_file=str(sample_csv), output_dir=str(out_dir))

    combined = pd.concat([
        pd.read_csv(out_dir / "positive.csv"),
        pd.read_csv(out_dir / "negative.csv"),
        pd.read_csv(out_dir / "neutral.csv")
    ])

    assert ((combined["confidence"] >= 0.0) & (combined["confidence"] <= 1.0)).all()


def test_11_invariant_count_equality(tmp_path, sample_csv):
    out_dir = tmp_path / "out"
    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(sample_csv),
        output_dir=str(out_dir)
    )

    assert (
        stats["positive_count"] + stats["negative_count"] + stats["neutral_count"]
        == stats["total_processed"]
    )
    assert stats["total_processed"] == 4


def test_12_empty_input_csv(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["id", "comment", "brand"]).to_csv(empty_csv, index=False)

    out_dir = tmp_path / "out"
    df_pos, df_neg, df_neu, stats = process_and_segregate_csv(
        input_file=str(empty_csv),
        output_dir=str(out_dir)
    )

    assert stats["total_processed"] == 0
    assert (out_dir / "positive.csv").exists()
    assert (out_dir / "negative.csv").exists()
    assert (out_dir / "neutral.csv").exists()
    assert len(pd.read_csv(out_dir / "positive.csv")) == 0
