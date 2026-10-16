import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.clustering.embeddings import embedding_generator


def _load_csv(path):
    df = pd.read_csv(path)
    if "text" in df.columns and "comment" not in df.columns:
        df = df.rename(columns={"text": "comment"})
    return df


def evaluate(input_path, k_min=3, k_max=10, out_dir="results"):
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.csv"))
        dfs = [_load_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _load_csv(input_path)

    required = ["platform", "brand", "comment", "sentiment"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    neg = df["sentiment"].str.lower() == "negative"
    comments = df.loc[neg, "comment"].dropna().tolist()
    print(f"Total rows: {len(df)}  |  Negative comments: {len(comments)}")

    if len(comments) < k_max:
        raise ValueError(f"Only {len(comments)} negative comments — need at least {k_max} for the largest K.")

    print("Encoding comments with sentence-transformers (this may take a moment)...")
    emb = embedding_generator.encode(comments)
    print(f"Embeddings shape: {emb.shape}")

    records = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb)
        sil = silhouette_score(emb, labels)
        inertia = km.inertia_
        sizes = np.bincount(labels).tolist()
        records.append({"k": k, "silhouette": round(sil, 4), "inertia": round(inertia, 2), "cluster_sizes": sizes})
        print(f"  k={k}  silhouette={sil:.4f}  inertia={inertia:.1f}  sizes={sizes}")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "k_selection_embeddings.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    ks = [r["k"] for r in records]
    sils = [r["silhouette"] for r in records]
    inertias = [r["inertia"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, inertias, marker="o", linewidth=2)
    ax1.set_title("Elbow Method (Inertia vs K)")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, sils, marker="s", linewidth=2, color="green")
    ax2.set_title("Silhouette Score vs K")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "k_selection_embeddings.png")
    plt.savefig(png_path, dpi=150)
    print(f"Plot saved to {png_path}")

    best = max(records, key=lambda r: r["silhouette"])
    print(f"\n>>> Best K by silhouette: k={best['k']} (silhouette={best['silhouette']})")

    return records


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_embeddings_clustering.py <csv_file_or_folder> [k_min] [k_max] [out_dir]")
        sys.exit(1)
    inp = sys.argv[1]
    kmin = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    kmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    outd = sys.argv[4] if len(sys.argv) > 4 else "results"
    evaluate(inp, kmin, kmax, outd)
