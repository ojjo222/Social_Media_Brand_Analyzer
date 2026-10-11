import logging
from typing import Optional, Union
from src.config import settings
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer

logger = logging.getLogger(__name__)


def get_sentiment_analyzer(
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Union[DistilBertSentimentAnalyzer, SentimentAnalyzer]:
    selected = (model_name or getattr(settings, "SENTIMENT_MODEL", "distilbert")).lower()

    if selected in ["distilbert", "transformer", "bert", "distilbert_v2", "v2", "distilbert_v1", "v1"]:
        if model_dir:
            candidate_dirs = [model_dir]
        elif selected in ["distilbert_v1", "v1"]:
            candidate_dirs = ["models/distilbert/gfg_distilbert_sentiment"]
        elif selected in ["distilbert_v2", "v2"]:
            candidate_dirs = ["models/distilbert/gfg_distilbert_sentiment_v2"]
        else:
            # Preferred order: v2 first, then v1 fallback, then base models/distilbert
            candidate_dirs = [
                "models/distilbert/gfg_distilbert_sentiment_v2",
                "models/distilbert/gfg_distilbert_sentiment",
                "models/distilbert"
            ]

        for target_dir in candidate_dirs:
            try:
                analyzer = DistilBertSentimentAnalyzer(model_dir=target_dir, device=device)
                if analyzer.is_loaded():
                    logger.info("Loaded DistilBERT analyzer from %s (version: %s)", analyzer.model_path, analyzer.version)
                    return analyzer
            except Exception as e:
                logger.warning("Failed to initialize DistilBERT at %s (%s).", target_dir, e)

        logger.warning("No DistilBERT model artifacts could be loaded. Falling back to TF-IDF.")

    target_dir = model_dir if (model_dir and selected in ["tfidf", "logistic_regression"]) else "models/sentiment"
    return SentimentAnalyzer(model_dir=target_dir)
