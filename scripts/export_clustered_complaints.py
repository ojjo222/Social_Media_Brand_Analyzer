import os, sys, json, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering.clusterer import ComplaintClusterer
from src.clustering.embeddings import embedding_generator

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'comment'})
    return df

def export_clustered(csv_path: str, out_path: str, method: str = "kmeans", **kwargs):
    df = load_data(csv_path)
    texts = df["comment"].astype(str).tolist()
    clusterer = ComplaintClusterer()
    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 7)
        clusterer.train_embeddings(texts, n_clusters=n_clusters)
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.4)
        min_samples = kwargs.get("min_samples", 5)
        clusterer.train_dbscan(texts, eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unsupported method: {method}")
    predictions = []
    for txt in texts:
        predictions.append(clusterer.predict(txt))
    pred_df = pd.DataFrame(predictions)
    if 'complaint_category' in df.columns:
        df = df.drop(columns=['complaint_category'])
    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    result_df['cluster'] = result_df['complaint_category']
    final_cols = ['platform', 'brand', 'comment', 'sentiment', 'cluster']
    result_df = result_df[final_cols]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Exported {len(result_df)} rows to {out_path}")
    print(f"Columns: {result_df.columns.tolist()}")

if __name__ == "__main__":
    data_file = os.path.join('data', 'synthetic', 'comments.csv')
    out_file = os.path.join('data', 'processed', 'clustered_comments.csv')
    export_clustered(data_file, out_file, method="kmeans", n_clusters=7)
