
import pandas as pd

# Load the BERTopic output
df = pd.read_csv("bertopic_results.csv")
summary = pd.read_csv("topic_summary.csv")

print("\nTOPIC SUMMARY")
print(summary.to_string(index=False))

print("\nCOMMENTS GROUPED BY TOPIC")

for topic in sorted(df["discovered_topic"].unique()):
    print(f"\nTopic {topic}")

    comments = df[
        df["discovered_topic"] == topic
    ]["comment"]

    for comment in comments:
        print("-", comment)
