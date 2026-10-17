import os, sys, json, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering.clusterer import ComplaintClusterer

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' in df.columns and 'comment' not in df.columns:
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

def export_processed_splits(negative_path: str, positive_path: str, neutral_path: str, out_path: str, n_clusters: int = 7):
    df_neg = load_data(negative_path)
    texts = df_neg["comment"].astype(str).tolist()
    
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(texts, n_clusters=n_clusters)
    
    labels = clusterer.kmeans.labels_
    cluster_themes = [clusterer.cluster_themes.get(str(int(lbl)), f"cluster_{lbl}") for lbl in labels]
    df_neg['cluster'] = cluster_themes
    
    df_pos = load_data(positive_path)
    df_pos['cluster'] = "None"
    
    df_neu = load_data(neutral_path)
    df_neu['cluster'] = "None"
    
    final_cols = ['platform', 'brand', 'comment', 'sentiment', 'cluster']
    df_neg = df_neg[final_cols]
    df_pos = df_pos[final_cols]
    df_neu = df_neu[final_cols]
    
    result_df = pd.concat([df_neg, df_pos, df_neu], ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Exported {len(result_df)} rows to {out_path}")
    print(f"Columns: {result_df.columns.tolist()}")

if __name__ == "__main__":
    neg_file = os.path.join('data', 'processed', 'negative.csv')
    pos_file = os.path.join('data', 'processed', 'positive.csv')
    neu_file = os.path.join('data', 'processed', 'neutral.csv')
    out_file = os.path.join('data', 'processed', 'clustered_comments.csv')

    if os.path.exists(neg_file) and os.path.exists(pos_file) and os.path.exists(neu_file):
        export_processed_splits(neg_file, pos_file, neu_file, out_file, n_clusters=7)
    else:
        data_file = os.path.join('data', 'synthetic', 'comments.csv')
        export_clustered(data_file, out_file, method="kmeans", n_clusters=7)
