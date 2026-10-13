
import pandas as pd

# Load BERTopic results
df = pd.read_csv("bertopic_results.csv")
topics = pd.read_csv("topic_summary.csv")

# Remove outliers for the topic summary
valid_topics = topics[topics["Topic"] != -1]

print("\nDiscovered Topics:")
print(valid_topics.to_string(index=False))

# Create readable topic labels
topic_names = {}

for _, row in valid_topics.iterrows():
    topic_id = row["Topic"]
    topic_name = str(row["Name"])
    topic_names[topic_id] = topic_name

# Add topic names to comments
df["topic_name"] = df["discovered_topic"].map(topic_names)

# Label outliers
df["topic_name"] = df["topic_name"].fillna(
    "Unassigned / Outlier"
)

# Save integrated results
df.to_csv("integrated_results.csv", index=False)

print("\nSaved integrated_results.csv")
print("\nSample Results:")
print(
    df[["comment", "discovered_topic", "topic_name"]]
    .to_string(index=False)
)
