import pytest
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer, clean_text_for_transformer


def test_clean_text_for_transformer():
    assert clean_text_for_transformer(None) == ""
    assert clean_text_for_transformer("   ") == ""
    assert clean_text_for_transformer("Check https://example.com/item &amp; deal") == "Check & deal"
    assert clean_text_for_transformer("I love this service!") == "I love this service!"


def test_distilbert_analyzer_loaded():
    analyzer = DistilBertSentimentAnalyzer()
    assert analyzer.is_loaded() is True
    assert analyzer.device.type in ["cpu", "cuda"]
    assert analyzer.id2label[0] == "negative"
    assert analyzer.id2label[1] == "neutral"
    assert analyzer.id2label[2] == "positive"


def test_distilbert_v2_preferred_by_default():
    analyzer = DistilBertSentimentAnalyzer()
    assert analyzer.is_loaded() is True
    assert analyzer.version == "v2"
    assert "gfg_distilbert_sentiment_v2" in analyzer.model_path


def test_distilbert_v1_explicit_loading():
    analyzer = DistilBertSentimentAnalyzer(model_dir="models/distilbert/gfg_distilbert_sentiment")
    assert analyzer.is_loaded() is True
    assert analyzer.version == "v1"
    assert "gfg_distilbert_sentiment" in analyzer.model_path
    assert analyzer.id2label[0] == "negative"
    assert analyzer.id2label[1] == "neutral"
    assert analyzer.id2label[2] == "positive"


def test_distilbert_predict_positive():
    analyzer = DistilBertSentimentAnalyzer()
    res = analyzer.predict("I absolutely love this product! Super fast delivery and amazing quality.")
    assert "sentiment" in res
    assert "confidence" in res
    assert res["sentiment"] == "positive"
    assert 0.0 <= res["confidence"] <= 1.0


def test_distilbert_predict_negative():
    analyzer = DistilBertSentimentAnalyzer()
    res = analyzer.predict("Tracking says order is delayed by 5 days. Terrible service and rude staff.")
    assert "sentiment" in res
    assert "confidence" in res
    assert res["sentiment"] == "negative"
    assert 0.0 <= res["confidence"] <= 1.0


def test_distilbert_predict_neutral():
    analyzer = DistilBertSentimentAnalyzer()
    res = analyzer.predict("What is the return window for this order?")
    assert "sentiment" in res
    assert "confidence" in res
    assert res["sentiment"] == "neutral"
    assert 0.0 <= res["confidence"] <= 1.0


def test_distilbert_predict_batch():
    analyzer = DistilBertSentimentAnalyzer()
    texts = [
        "Highly recommended! Outstanding customer support.",
        "Worst app ever, payment failed and account debited.",
        "When does the sale start tomorrow?",
        ""
    ]
    results = analyzer.predict_batch(texts)
    assert len(results) == 4
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"
    assert results[2]["sentiment"] == "neutral"
    assert 0.0 <= results[0]["confidence"] <= 1.0
    assert 0.0 <= results[1]["confidence"] <= 1.0
    assert 0.0 <= results[2]["confidence"] <= 1.0


def test_distilbert_predict_batch_empty():
    analyzer = DistilBertSentimentAnalyzer()
    results = analyzer.predict_batch([])
    assert results == []


def test_distilbert_edge_cases():
    analyzer = DistilBertSentimentAnalyzer()
    res_empty = analyzer.predict("")
    assert res_empty["sentiment"] == "neutral"

    res_short = analyzer.predict("ok")
    assert "sentiment" in res_short
    assert "confidence" in res_short


def test_distilbert_cpu_explicit_device():
    analyzer = DistilBertSentimentAnalyzer(device="cpu")
    assert analyzer.device.type == "cpu"
    res = analyzer.predict("Great service!")
    assert res["sentiment"] == "positive"


def test_distilbert_unloaded_error():
    analyzer = DistilBertSentimentAnalyzer(model_dir="non_existent_distilbert_dir_123")
    assert analyzer.is_loaded() is False
    with pytest.raises(RuntimeError):
        analyzer.predict("Test")
    with pytest.raises(RuntimeError):
        analyzer.predict_batch(["Test"])
