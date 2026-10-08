import pandas as pd
from datetime import datetime
import os

df = pd.read_csv("final_dataset_english.csv", encoding="utf-8")
print(f"Total rows: {len(df)}")

today = datetime.now().strftime("%d%b").lower()
output_dir = f"delivery_english_{today}"
os.makedirs(output_dir, exist_ok=True)

batch_size = 10000
n_batches = (len(df) + batch_size - 1) // batch_size

for i in range(n_batches):
    chunk = df.iloc[i * batch_size : (i + 1) * batch_size]
    chunk.to_csv(f"{output_dir}/comments_batch_{i+1:02d}.csv", index=False, encoding="utf-8")

print(f"Split into {n_batches} batches in {output_dir}/")