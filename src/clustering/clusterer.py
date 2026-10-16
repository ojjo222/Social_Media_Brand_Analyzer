import os
import json
import joblib
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, DBSCAN

try:
    from src.clustering.embeddings import embedding_generator
    from src.preprocessing.cleaner import preprocess_for_ml
except ImportError:
    def preprocess_for_ml(text: str) -> str:
        return str(text).lower()

DEFAULT_BRAND_STOPWORDS = {
    "amazon", "flipkart", "swiggy", "zomato", "phonepe", "paytm",
    "twitter", "instagram", "facebook", "reddit"
}

THEME_KEYWORDS = {
    "Delivery Problems": ["delivery", "late", "tracking", "delivered", "order", "package"],
    "Customer Support": ["customer", "support", "service", "hold", "bot", "human", "agent", "reply"],
    "Billing & Overcharging": ["charged", "overcharged", "twice", "billing", "fee", "invoice", "renew"],
    "Refund Issues": ["refund", "money", "bank", "reflecting", "return"],
    "App & Technical Bugs": ["app", "crash", "bug", "buggy", "login", "checkout", "error", "website", "pay"],
    "Product Quality": ["quality", "broken", "cold", "stale", "packaging", "item", "food", "damaged"]
}


def infer_theme_category(keywords: str) -> str:
    tokens = set(keywords.lower().split())
    best_theme = None
    max_matches = 0
    for theme, kw_list in THEME_KEYWORDS.items():
        matches = sum(1 for kw in kw_list if kw in tokens or any(kw in t for t in tokens))
        if matches > max_matches:
            max_matches = matches
            best_theme = theme
    return best_theme if best_theme else keywords.title()


class ComplaintClusterer:
    def __init__(self, model_dir: str = "models/clustering"):
        self.model_dir = model_dir
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Dict[str, str] = {}
        self.cluster_themes: Dict[str, str] = {}
        if (
            os.path.exists(os.path.join(model_dir, "kmeans.joblib"))
            and os.path.exists(os.path.join(model_dir, "vectorizer.joblib"))
            and os.path.exists(os.path.join(model_dir, "cluster_labels.json"))
        ):
            self.load(model_dir)

    def train(self, texts: List[str], n_clusters: int = 6):
        if not texts or len(texts) < n_clusters:
            raise ValueError(f"Texts list must contain at least {n_clusters} samples.")

        processed = [preprocess_for_ml(t) for t in texts]
        combined_stops = list(ENGLISH_STOP_WORDS.union(DEFAULT_BRAND_STOPWORDS))

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=combined_stops
        )
        X = self.vectorizer.fit_transform(processed)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X)
        self._generate_cluster_labels(n_clusters)

    def _generate_cluster_labels(self, n_clusters: int, top_n: int = 3):
        terms = self.vectorizer.get_feature_names_out()
        centroids = self.kmeans.cluster_centers_
        self.cluster_labels = {}
        self.cluster_themes = {}

        for i in range(n_clusters):
            top_term_indices = centroids[i].argsort()[-top_n:][::-1]
            top_terms = [terms[ind] for ind in top_term_indices]
            keywords = " ".join(top_terms)
            self.cluster_labels[str(i)] = keywords
            self.cluster_themes[str(i)] = infer_theme_category(keywords)

    def _generate_embedding_cluster_labels(self, texts: List[str], cluster_ids: List[int], n_clusters: int, top_n: int = 3):
        self.cluster_labels = {}
        self.cluster_themes = {}
        for i in range(n_clusters):
            c_texts = [texts[idx] for idx, cid in enumerate(cluster_ids) if cid == i]
            if c_texts:
                try:
                    c_vec = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
                    X = c_vec.fit_transform(c_texts)
                    mean_scores = X.mean(axis=0).A1
                    top_idx = mean_scores.argsort()[-top_n:][::-1]
                    terms = c_vec.get_feature_names_out()
                    keywords = " ".join([terms[idx] for idx in top_idx if idx < len(terms)])
                except Exception:
                    keywords = f"cluster_{i}"
            else:
                keywords = f"cluster_{i}"
            if not keywords.strip():
                keywords = f"cluster_{i}"
            self.cluster_labels[str(i)] = keywords
            self.cluster_themes[str(i)] = infer_theme_category(keywords)

    def train_embeddings(self, texts: List[str], n_clusters: int = 6):
        if not texts or len(texts) < n_clusters:
            raise ValueError(f"Texts list must contain at least {n_clusters} samples.")
        self.vectorizer = None
        emb = embedding_generator.encode(texts)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = self.kmeans.fit_predict(emb)
        self._generate_embedding_cluster_labels(texts, cluster_ids.tolist(), n_clusters)

    def train_dbscan(self, texts: List[str], eps: float = 0.5, min_samples: int = 5):
        if not texts:
            raise ValueError("Texts list must not be empty.")
        self.vectorizer = None
        emb = embedding_generator.encode(texts)
        self.kmeans = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_ids = self.kmeans.fit_predict(emb)
        unique_labels = sorted([lbl for lbl in set(cluster_ids) if lbl != -1])
        self.cluster_labels = {}
        self.cluster_themes = {}
        for lbl in unique_labels:
            c_texts = [texts[idx] for idx, cid in enumerate(cluster_ids) if cid == lbl]
            if c_texts:
                try:
                    c_vec = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
                    X = c_vec.fit_transform(c_texts)
                    mean_scores = X.mean(axis=0).A1
                    top_idx = mean_scores.argsort()[-3:][::-1]
                    terms = c_vec.get_feature_names_out()
                    keywords = " ".join([terms[idx] for idx in top_idx if idx < len(terms)])
                except Exception:
                    keywords = f"cluster_{lbl}"
            else:
                keywords = f"cluster_{lbl}"
            if not keywords.strip():
                keywords = f"cluster_{lbl}"
            self.cluster_labels[str(lbl)] = keywords
            self.cluster_themes[str(lbl)] = infer_theme_category(keywords)

    def predict_embeddings(self, text: str) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        emb = embedding_generator.encode([text])
        cluster_id = int(self.kmeans.predict(emb)[0])
        cid_str = str(cluster_id)
        return {"cluster_id": cluster_id, "cluster_label": self.cluster_labels.get(cid_str, cid_str), "complaint_category": self.cluster_themes.get(cid_str, self.cluster_labels.get(cid_str, cid_str))}

    def predict_batch_embeddings(self, texts: List[str]) -> list:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        embs = embedding_generator.encode(texts)
        try:
            cluster_ids = self.kmeans.predict(embs)
        except AttributeError:
            cluster_ids = self.kmeans.fit_predict(embs)
        results = []
        for cid in cluster_ids:
            cid_str = str(int(cid))
            results.append({"cluster_id": int(cid), "cluster_label": self.cluster_labels.get(cid_str, cid_str), "complaint_category": self.cluster_themes.get(cid_str, self.cluster_labels.get(cid_str, cid_str))})
        return results

    def predict(self, text: str) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        if self.vectorizer is not None:
            processed = preprocess_for_ml(text)
            X = self.vectorizer.transform([processed])
            try:
                cluster_id = int(self.kmeans.predict(X)[0])
            except AttributeError:
                cluster_id = int(self.kmeans.fit_predict(X)[0])
        else:
            emb = embedding_generator.encode([text])
            try:
                cluster_id = int(self.kmeans.predict(emb)[0])
            except AttributeError:
                cluster_id = int(self.kmeans.fit_predict(emb)[0])
        cid_str = str(cluster_id)
        return {
            "cluster_id": cluster_id,
            "cluster_label": self.cluster_labels.get(cid_str, cid_str),
            "complaint_category": self.cluster_themes.get(cid_str, self.cluster_labels.get(cid_str, cid_str))
        }

    def predict_batch(self, texts: List[str]) -> list:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        processed = [preprocess_for_ml(t) for t in texts]
        X = self.vectorizer.transform(processed)
        try:
            cluster_ids = self.kmeans.predict(X)
        except AttributeError:
            cluster_ids = self.kmeans.fit_predict(X)
        results = []
        for cid in cluster_ids:
            cid_str = str(int(cid))
            results.append({
                "cluster_id": int(cid),
                "cluster_label": self.cluster_labels.get(cid_str, cid_str),
                "complaint_category": self.cluster_themes.get(cid_str, self.cluster_labels.get(cid_str, cid_str))
            })
        return results

    def get_cluster_labels(self) -> dict:
        return self.cluster_labels

    def get_cluster_themes(self) -> dict:
        return self.cluster_themes

    def save(self, model_dir: Optional[str] = None):
        if not self.is_loaded():
            raise RuntimeError("No model to save.")
        dir_to_save = model_dir or self.model_dir
        os.makedirs(dir_to_save, exist_ok=True)
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, os.path.join(dir_to_save, "vectorizer.joblib"))
        joblib.dump(self.kmeans, os.path.join(dir_to_save, "kmeans.joblib"))
        with open(os.path.join(dir_to_save, "cluster_labels.json"), "w", encoding="utf-8") as f:
            json.dump(self.cluster_labels, f, indent=4)
        with open(os.path.join(dir_to_save, "cluster_themes.json"), "w", encoding="utf-8") as f:
            json.dump(self.cluster_themes, f, indent=4)

    def load(self, model_dir: Optional[str] = None):
        dir_to_load = model_dir or self.model_dir
        self.vectorizer = joblib.load(os.path.join(dir_to_load, "vectorizer.joblib"))
        self.kmeans = joblib.load(os.path.join(dir_to_load, "kmeans.joblib"))
        labels_file = os.path.join(dir_to_load, "cluster_labels.json")
        if os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.cluster_labels = json.load(f)
        themes_file = os.path.join(dir_to_load, "cluster_themes.json")
        if os.path.exists(themes_file):
            with open(themes_file, "r", encoding="utf-8") as f:
                self.cluster_themes = json.load(f)
        else:
            self.cluster_themes = {
                cid: infer_theme_category(label)
                for cid, label in self.cluster_labels.items()
            }

    def is_loaded(self) -> bool:
        return self.kmeans is not None
