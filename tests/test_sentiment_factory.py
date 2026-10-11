import pytest
from src.sentiment.factory import get_sentiment_analyzer
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer


def test_factory_returns_distilbert_v2_by_default():
    analyzer = get_sentiment_analyzer("distilbert")
    assert isinstance(analyzer, DistilBertSentimentAnalyzer)
    assert analyzer.is_loaded() is True
    assert analyzer.version == "v2"
    assert "gfg_distilbert_sentiment_v2" in analyzer.model_path


def test_factory_returns_distilbert_v1_explicit():
    analyzer = get_sentiment_analyzer("distilbert_v1")
    assert isinstance(analyzer, DistilBertSentimentAnalyzer)
    assert analyzer.is_loaded() is True
    assert analyzer.version == "v1"
    assert "gfg_distilbert_sentiment" in analyzer.model_path


def test_factory_returns_distilbert_v2_explicit():
    analyzer = get_sentiment_analyzer("distilbert_v2")
    assert isinstance(analyzer, DistilBertSentimentAnalyzer)
    assert analyzer.is_loaded() is True
    assert analyzer.version == "v2"


def test_factory_returns_tfidf_explicit():
    analyzer = get_sentiment_analyzer("tfidf")
    assert isinstance(analyzer, SentimentAnalyzer)
    assert analyzer.is_loaded() is True


def test_factory_fallback_on_invalid_distilbert_path():
    analyzer = get_sentiment_analyzer("distilbert", model_dir="non_existent_folder_abc123")
    assert isinstance(analyzer, SentimentAnalyzer)
    assert analyzer.is_loaded() is True


def test_factory_analyzers_interface_uniformity():
    v2 = get_sentiment_analyzer("distilbert_v2")
    v1 = get_sentiment_analyzer("distilbert_v1")
    tfidf = get_sentiment_analyzer("tfidf")

    text = "The product arrived broken and customer support was rude."
    res_v2 = v2.predict(text)
    res_v1 = v1.predict(text)
    res_t = tfidf.predict(text)

    for res in [res_v2, res_v1, res_t]:
        assert "sentiment" in res
        assert "confidence" in res
        assert res["sentiment"] in ["positive", "neutral", "negative"]
        assert 0.0 <= res["confidence"] <= 1.0

    batch_texts = ["Great experience!", "Awful delay and bad packaging."]
    b_res_v2 = v2.predict_batch(batch_texts)
    b_res_v1 = v1.predict_batch(batch_texts)
    b_res_t = tfidf.predict_batch(batch_texts)

    assert len(b_res_v2) == 2
    assert len(b_res_v1) == 2
    assert len(b_res_t) == 2
