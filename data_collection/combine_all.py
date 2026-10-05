import pandas as pd
from pathlib import Path
from datetime import datetime
import os

today = datetime.now().strftime("%d%b").lower()
output_dir = f"delivery_{today}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output folder: {output_dir}/")

print("Loading sources...")

yt_real = pd.read_csv("kaggle_clean.csv", encoding="utf-8")
yt_real["platform"] = "youtube"

tw_real = pd.read_csv("twitter_clean.csv", encoding="utf-8")
tw_real["platform"] = "twitter"


def load_synth(path):
    if not Path(path).exists():
        print(f"  {path} not found. Skipping.")
        return pd.DataFrame(columns=["text", "sentiment", "brand", "platform", "source"])
    return pd.read_csv(path, encoding="utf-8")


syn_gem = load_synth("synthetic_gemini.csv")
syn_grq = load_synth("synthetic_groq.csv")

if "brand" not in yt_real.columns:
    yt_real["brand"] = "unknown"


def normalize(df):
    keep = ["text", "sentiment", "brand", "platform", "source"]
    for col in keep:
        if col not in df.columns:
            df[col] = "unknown"
    return df[keep]


yt_real = normalize(yt_real)
tw_real = normalize(tw_real)
syn_gem = normalize(syn_gem)
syn_grq = normalize(syn_grq)

df = pd.concat([yt_real, tw_real, syn_gem, syn_grq], ignore_index=True)
print(f"Combined: {len(df)} rows")

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 3]
df = df.drop_duplicates(subset=["text"])
print(f"After dedupe: {len(df)} rows")

print("\nBy platform:")
print(df["platform"].value_counts())
print("\nBy sentiment:")
print(df["sentiment"].value_counts())
print("\nBy source:")
print(df["source"].value_counts())

df.to_csv("final_dataset.csv", index=False, encoding="utf-8")
print(f"\nSaved final_dataset.csv with {len(df)} rows")

batch_size = 10000
n_batches = (len(df) + batch_size - 1) // batch_size
for i in range(n_batches):
    chunk = df.iloc[i * batch_size : (i + 1) * batch_size]
    chunk.to_csv(f"{output_dir}/comments_batch_{i+1:02d}.csv", index=False, encoding="utf-8")

print(f"\nSaved {n_batches} batch files inside {output_dir}/")
print(f"Done.")