import pandas as pd

df = pd.read_csv("YoutubeCommentsDataSet.csv")

print(f"Before: {len(df)} rows")

df = df.dropna(subset=["Comment"])

df = df.drop_duplicates(subset=["Comment"])

df["Sentiment"] = df["Sentiment"].str.lower().str.strip()

df = df[df["Sentiment"].isin(["positive", "negative", "neutral"])]

df = df.rename(columns={"Comment": "text", "Sentiment": "sentiment"})

df["source"] = "kaggle"

df.to_csv("kaggle_clean.csv", index=False)

print(f"After:  {len(df)} rows")
print(f"\nSentiment breakdown:")
print(df["sentiment"].value_counts())
print(f"\nSaved to kaggle_clean.csv")