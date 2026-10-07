import pandas as pd
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

print("Loading dataset...")
df = pd.read_csv("final_dataset.csv", encoding="utf-8")
print(f"Rows before: {len(df)}")

print("Detecting languages (this will take 10-15 min)...")

def safe_detect(text):
    try:
        if len(str(text)) > 15:
            return detect(str(text))
        return "short"
    except Exception:
        return "unknown"

df["language"] = df["text"].apply(safe_detect)
print("Detection done.")

print("\nLanguage breakdown (top 20):")
print(df["language"].value_counts().head(20))

# Keep English + short (short ones are usually English fragments)
df_en = df[df["language"].isin(["en", "short"])].copy()
df_en = df_en.drop(columns=["language"])

print(f"\nRows after filtering: {len(df_en)}")

# Save
df_en.to_csv("final_dataset_english.csv", index=False, encoding="utf-8")
print("Saved final_dataset_english.csv")