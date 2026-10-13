
import pandas as pd
from bertopic import BERTopic

# Read the data
df = pd.read_csv("sample_clustering_results.csv")

# Select negative comments
negative_df = df[
    df["sentiment"].str.lower() == "negative"
].copy()

# Get comment text
comments = negative_df["comment"].dropna().astype(str).tolist()

# Check whether enough comments are available
if len(comments) < 10:
    print("Please collect more comments for topic modeling.")
else:
    # Create the topic model
    topic_model = BERTopic(
        min_topic_size=3,
        verbose=True
    )

    # Discover topics
    topics, probabilities = topic_model.fit_transform(comments)

    # Store discovered topic numbers
    negative_df = negative_df.loc[
        negative_df["comment"].notna()
    ].copy()

    negative_df["discovered_topic"] = topics

    # Display topic information
    print("\nDiscovered Topics:")
    print(topic_model.get_topic_info())

    # Display comments and their topics
    print("\nComments with Discovered Topics:")
    print(
        negative_df[
            ["comment", "discovered_topic"]
        ].to_string(index=False)
    )

    # Save results
    negative_df.to_csv(
        "bertopic_results.csv",
        index=False
    )

    # Save topic descriptions
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("topic_summary.csv", index=False)

    print("\nSaved bertopic_results.csv")
    print("Saved topic_summary.csv")

