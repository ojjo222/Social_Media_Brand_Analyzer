import os, sys, json, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.clustering.clusterer import ComplaintClusterer
from sklearn.metrics import silhouette_score

def load_data(path):
    df = pd.read_csv(path)
    return df[df['sentiment'].str.lower() == 'negative']['comment'].tolist()

def eval_kmeans(texts, n_clusters=7):
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(texts, n_clusters=n_clusters)
    from src.clustering.embeddings import embedding_generator
    emb = embedding_generator.encode(texts)
    score = silhouette_score(emb, clusterer.kmeans.labels_)
    return {"method": "kmeans", "n_clusters": n_clusters, "silhouette": score, "noise_ratio": 0.0}

def eval_dbscan(texts, eps=0.4, min_samples=5):
    clusterer = ComplaintClusterer()
    clusterer.train_dbscan(texts, eps=eps, min_samples=min_samples)
    labels = clusterer.kmeans.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = list(labels).count(-1)
    noise_ratio = noise / len(labels)
    if n_clusters > 1:
        from src.clustering.embeddings import embedding_generator
        emb = embedding_generator.encode(texts)
        mask = labels != -1
        score = silhouette_score(emb[mask], labels[mask])
    else:
        score = None
    return {"method": "dbscan", "n_clusters": n_clusters, "silhouette": score, "noise_ratio": noise_ratio}

def compare(data_path, out_path):
    texts = load_data(data_path)
    results = []
    results.append(eval_kmeans(texts, n_clusters=7))
    results.append(eval_dbscan(texts, eps=0.4, min_samples=5))
    summary = {r['method']: {k: v for k, v in r.items() if k != 'method'} for r in results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    return summary

if __name__ == '__main__':
    data_file = os.path.join('data', 'synthetic', 'comments.csv')
    out_file = os.path.join('reports', 'clustering_comparison.json')
    print(compare(data_file, out_file))
