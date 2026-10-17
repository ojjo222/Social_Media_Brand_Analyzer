
import pandas as pd
from pathlib import Path

# Input and output files
results_file = Path("bertopic_results.csv")
summary_file = Path("topic_summary.csv")
output_file = Path("integrated_results.csv")

# Check whether input files exist
if not results_file.exists():
    print("Error: bertopic_results.csv not found.")
    raise SystemExit(1)

if not summary_file.exists():
    print("Error: topic_summary.csv not found.")
    raise SystemExit(1)

# Load BERTopic results
df = pd.read_csv(results_file)
topics = pd.read_csv(summary_file)

# Check required columns
if "discovered_topic" not in df.columns:
    print("Error: discovered_topic column is missing.")
    raise SystemExit(1)

if "Topic" not in topics.columns or "Name" not in topics.columns:
    print("Error: Topic or Name column is missing from topic_summary.csv.")
    raise SystemExit(1)

# Convert topic IDs to numeric values
df["discovered_topic"] = pd.to_numeric(
    df["discovered_topic"], errors="coerce"
)

topics["Topic"] = pd.to_numeric(
    topics["Topic"], errors="coerce"
)

# Remove invalid topic IDs and outliers from the topic-name mapping
valid_topics = topics[
    topics["Topic"].notna()
    & (topics["Topic"] != -1)
    & topics["Name"].notna()
].copy()

print("\nDiscovered Topics:")
print(valid_topics.to_string(index=False))

# Create topic ID-to-name mapping
topic_names = dict(
    zip(valid_topics["Topic"], valid_topics["Name"].astype(str))
)

# Add readable topic names to comments
df["topic_name"] = df["discovered_topic"].map(topic_names)

# Label outliers and missing topic names
df["topic_name"] = df["topic_name"].fillna(
    "Unassigned / Outlier"
)

# Save integrated results
df.to_csv(output_file, index=False)

# Display sample results
print("\nSaved integrated_results.csv")
print("Total comments:", len(df))

print("\nTopic counts:")
print(df["topic_name"].value_counts().to_string())

print("\nSample Results:")
print(
    df[["comment", "discovered_topic", "topic_name"]]
    .head(10)
    .to_string(index=False)
)
