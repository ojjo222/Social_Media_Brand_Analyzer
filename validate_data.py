
import pandas as pd
from pathlib import Path

FILE = Path("clustered_comments.csv")

if not FILE.exists():
    raise SystemExit("Error: clustered_comments.csv not found.")

df = pd.read_csv(FILE)

required = [
    "platform", "brand", "comment", "sentiment", "cluster"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise SystemExit(f"Missing columns: {missing}")

print("\nDATA VALIDATION")
print("-" * 40)
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing values:")
print(df[required].isna().sum().to_string())

print("\nSentiment counts:")
print(
    df["sentiment"]
    .astype("string")
    .str.strip()
    .str.lower()
    .value_counts(dropna=False)
    .to_string()
)

print("\nRows by cluster:")
print(
    df["cluster"]
    .fillna("Unclustered")
    .astype(str)
    .value_counts()
    .to_string()
)

print("\nDuplicate comments:", df["comment"].duplicated().sum())

print("\nValidation complete.")
