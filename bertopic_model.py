
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

# File paths
input_file = Path("clustered_comments.csv")
results_file = Path("bertopic_results.csv")
summary_file = Path("topic_summary.csv")

# Check whether the input file exists
if not input_file.exists():
    print("Error: clustered_comments.csv was not found.")
    print("Place the CSV file in the same folder as this Python file.")
    raise SystemExit

# Read the data
df = pd.read_csv(input_file)

# Check required columns
required_columns = ["comment", "sentiment"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("Error: Missing columns:", missing_columns)
    raise SystemExit

# Select negative comments safely
negative_df = df[
    df["sentiment"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("negative")
].copy()

# Remove missing or empty comments
negative_df = negative_df[
    negative_df["comment"].notna()
].copy()

negative_df["comment"] = negative_df["comment"].astype(str).str.strip()

negative_df = negative_df[
    negative_df["comment"] != ""
].copy()

# Check the number of comments
comments = negative_df["comment"].tolist()

print("Total comments:", len(df))
print("Negative comments for topic modeling:", len(comments))

if len(comments) < 10:
    print("Please collect at least 10 valid negative comments.")
    raise SystemExit

# Remove duplicate comments for more useful topic discovery
# Keep the original rows for saving results later
unique_comments = list(dict.fromkeys(comments))

if len(unique_comments) < 10:
    print("Not enough unique negative comments for topic modeling.")
    raise SystemExit

# Create the BERTopic model
topic_model = BERTopic(
    min_topic_size=10,
    verbose=True
)

# Discover topics using unique comments
topics, probabilities = topic_model.fit_transform(unique_comments)

# Map discovered topics back to all negative comments
topic_lookup = dict(zip(unique_comments, topics))
negative_df["discovered_topic"] = negative_df["comment"].map(topic_lookup)

# Display topic information
topic_info = topic_model.get_topic_info()

print("\nDiscovered Topics:")
print(topic_info.to_string(index=False))

# Display comments and their discovered topics
print("\nComments with Discovered Topics:")
print(
    negative_df[
        ["comment", "discovered_topic"]
    ].to_string(index=False)
)

# Save negative comments with discovered topic IDs
negative_df.to_csv(results_file, index=False)

# Save topic summaries and representative keywords
topic_info.to_csv(summary_file, index=False)

print("\nFiles saved successfully:")
print("- bertopic_results.csv")
print("- topic_summary.csv")

print("\nNote: Topic ID -1 means BERTopic marked a comment as an outlier.")

