import pandas as pd

df = pd.read_csv("YoutubeCommentsDataSet.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nNulls:")
print(df.isnull().sum())
print("\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique")