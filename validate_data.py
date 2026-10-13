
import pandas as pd

# File to check
file_name = "sample_clustering_results.csv"

# Required columns
required_columns = [
    "platform",
    "brand",
    "comment",
    "sentiment",
    "cluster"
]

try:
    df = pd.read_csv(file_name)

    print("\nDATA VALIDATION REPORT")

    # Check columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("All required columns are present.")

    # Check empty comments
    if "comment" in df.columns:
        empty_comments = (
            df["comment"].isna()
            | df["comment"].astype(str).str.strip().eq("")
        )

        print(
            "Empty comments:",
            int(empty_comments.sum())
        )

    # Check sentiment labels
    if "sentiment" in df.columns:
        valid_sentiments = [
            "positive",
            "negative",
            "neutral"
        ]

        actual = (
            df["sentiment"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        invalid = ~actual.isin(valid_sentiments)

        print(
            "Invalid sentiment labels:",
            int(invalid.sum())
        )

    # Check duplicate rows
    print(
        "Duplicate rows:",
        int(df.duplicated().sum())
    )

    print("Total rows:", len(df))

except FileNotFoundError:
    print("CSV file not found:", file_name)

except pd.errors.ParserError:
    print("Could not read the CSV. Check its formatting.")
