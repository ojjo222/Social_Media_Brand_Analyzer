import pandas as pd

train = pd.read_csv("twitter_training.csv", header=None, names=["id", "entity", "sentiment", "text"])
valid = pd.read_csv("twitter_validation.csv", header=None, names=["id", "entity", "sentiment", "text"])

df = pd.concat([train, valid], ignore_index=True)
print(f"Combined: {len(df)} rows")

df = df.dropna(subset=["text"])

df = df.drop_duplicates(subset=["text"])
print(f"After dedupe: {len(df)} rows")

df["sentiment"] = df["sentiment"].str.lower().str.strip()

df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]
print(f"After dropping irrelevant: {len(df)} rows")

df = df.rename(columns={"entity": "brand"})[["text", "sentiment", "brand"]]

df["source"] = "twitter"

df.to_csv("twitter_clean.csv", index=False)
print(f"\n Saved twitter_clean.csv with {len(df)} rows")
print(f"\nSentiment breakdown:")
print(df["sentiment"].value_counts())
print(f"\nTop brands:")
print(df["brand"].value_counts().head(10))