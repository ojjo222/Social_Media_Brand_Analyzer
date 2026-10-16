import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.clustering.clusterer import ComplaintClusterer


TRAIN_TEXTS = [
    "My order never arrived",
    "Package delivery was delayed",
    "App crashes on login",
    "Login button does not respond",
    "Refund not processed",
    "Money not returned to account",
    "Customer support never replies",
    "No response from support team",
    "Charged extra fees",
    "Overcharged on my bill",
    "Tracking info not updated",
    "Billing error on invoice",
]


def _trained_tfidf_clusterer(n_clusters=3):
    clusterer = ComplaintClusterer()
    clusterer.train(TRAIN_TEXTS, n_clusters=n_clusters)
    return clusterer


def test_clusterer_loaded():
    clusterer = _trained_tfidf_clusterer()
    assert clusterer.is_loaded() is True
    assert len(clusterer.get_cluster_labels()) > 0
    assert len(clusterer.get_cluster_themes()) > 0


def test_clusterer_predict_single():
    clusterer = _trained_tfidf_clusterer()
    res = clusterer.predict("Still waiting for my refund after 2 weeks, where is my money?")
    assert "cluster_id" in res
    assert isinstance(res["cluster_id"], int)
    assert "cluster_label" in res
    assert "complaint_category" in res
    assert len(res["complaint_category"]) > 0


def test_clusterer_predict_batch():
    clusterer = _trained_tfidf_clusterer()
    texts = [
        "Delivery was 5 days late!",
        "Charged twice on my credit card",
        "App keeps crashing on checkout page"
    ]
    results = clusterer.predict_batch(texts)
    assert len(results) == 3
    for r in results:
        assert "cluster_id" in r
        assert "complaint_category" in r
        assert isinstance(r["cluster_id"], int)


def test_clusterer_edge_cases():
    clusterer = _trained_tfidf_clusterer()
    res_empty = clusterer.predict("")
    assert "cluster_id" in res_empty

    res_short = clusterer.predict("broken")
    assert "cluster_id" in res_short

    res_unknown = clusterer.predict("xyzabc random words 98765")
    assert "cluster_id" in res_unknown


def test_clusterer_unloaded_error():
    clusterer = ComplaintClusterer(model_dir="non_existent_clustering_dir_123")
    assert clusterer.is_loaded() is False
    with pytest.raises(RuntimeError):
        clusterer.predict("Some complaint")


FAKE_TEXTS = [
    "My order never arrived",
    "Package delivery was delayed",
    "App crashes on login",
    "Login button does not respond",
    "Refund not processed",
    "Money not returned to account",
    "Customer support never replies",
    "No response from support team",
    "Charged extra fees",
    "Overcharged on my bill",
]


def _make_fake_embeddings(n, dim=384):
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


@patch("src.clustering.clusterer.embedding_generator")
def test_train_embeddings_output_shape(mock_gen):
    mock_gen.encode.return_value = _make_fake_embeddings(len(FAKE_TEXTS))
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(FAKE_TEXTS, n_clusters=3)
    assert clusterer.kmeans is not None
    assert len(clusterer.cluster_labels) == 3
    assert len(clusterer.cluster_themes) == 3


@patch("src.clustering.clusterer.embedding_generator")
def test_train_embeddings_predict_returns_correct_keys(mock_gen):
    mock_gen.encode.return_value = _make_fake_embeddings(len(FAKE_TEXTS))
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(FAKE_TEXTS, n_clusters=3)

    single_emb = _make_fake_embeddings(1)
    mock_gen.encode.return_value = single_emb
    result = clusterer.predict("some complaint")
    assert "cluster_id" in result
    assert "cluster_label" in result
    assert "complaint_category" in result
    assert isinstance(result["cluster_id"], int)


@patch("src.clustering.clusterer.embedding_generator")
def test_train_embeddings_too_few_samples_raises(mock_gen):
    mock_gen.encode.return_value = _make_fake_embeddings(2)
    clusterer = ComplaintClusterer()
    with pytest.raises(ValueError):
        clusterer.train_embeddings(["text1", "text2"], n_clusters=5)


@patch("src.clustering.clusterer.embedding_generator")
def test_train_dbscan_produces_labels(mock_gen):
    mock_gen.encode.return_value = _make_fake_embeddings(len(FAKE_TEXTS))
    clusterer = ComplaintClusterer()
    clusterer.train_dbscan(FAKE_TEXTS, eps=0.9, min_samples=2)
    assert clusterer.kmeans is not None
    assert isinstance(clusterer.cluster_labels, dict)


@patch("src.clustering.clusterer.embedding_generator")
def test_train_dbscan_empty_raises(mock_gen):
    clusterer = ComplaintClusterer()
    with pytest.raises(ValueError):
        clusterer.train_dbscan([], eps=0.5, min_samples=5)


@patch("src.clustering.clusterer.embedding_generator")
def test_predict_batch_embeddings_length(mock_gen):
    mock_gen.encode.return_value = _make_fake_embeddings(len(FAKE_TEXTS))
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(FAKE_TEXTS, n_clusters=3)

    mock_gen.encode.return_value = _make_fake_embeddings(5)
    results = clusterer.predict_batch_embeddings(FAKE_TEXTS[:5])
    assert len(results) == 5
    for r in results:
        assert "cluster_id" in r
        assert "complaint_category" in r


@pytest.mark.integration
def test_integration_real_model_end_to_end():
    texts = [
        "My order is 5 days late",
        "Package never arrived",
        "App crashes on checkout",
        "Login button does not respond",
        "Refund not processed yet",
        "Money not returned",
        "Customer support silent",
        "No agent replied to my ticket",
        "Charged extra fees",
        "Overcharged on bill",
    ]
    clusterer = ComplaintClusterer()
    clusterer.train_embeddings(texts, n_clusters=3)
    result = clusterer.predict(texts[0])
    assert "cluster_id" in result
    assert isinstance(result["cluster_id"], int)
    assert "complaint_category" in result

