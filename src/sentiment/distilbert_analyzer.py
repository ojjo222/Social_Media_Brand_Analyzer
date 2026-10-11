import os
import html
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def clean_text_for_transformer(text: Union[str, None]) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DistilBertSentimentAnalyzer:
    def __init__(self, model_dir: str = "models/distilbert", device: Optional[str] = None):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.model_path = None
        self.version = "unknown"
        self.id2label = DEFAULT_ID2LABEL.copy()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load(model_dir)

    def _resolve_model_path(self, model_dir: str) -> Path:
        base_path = Path(model_dir)
        # 1. Direct path check
        if (base_path / "config.json").exists():
            return base_path
        # 2. Prefer v2 when present
        v2_path = base_path / "gfg_distilbert_sentiment_v2"
        if v2_path.exists() and (v2_path / "config.json").exists():
            return v2_path
        # 3. Fallback to v1
        v1_path = base_path / "gfg_distilbert_sentiment"
        if v1_path.exists() and (v1_path / "config.json").exists():
            return v1_path
        return base_path

    def load(self, model_dir: Optional[str] = None):
        load_dir = model_dir or self.model_dir
        resolved_path = self._resolve_model_path(load_dir)

        if not resolved_path.exists() or not (resolved_path / "config.json").exists():
            self.tokenizer = None
            self.model = None
            self.model_path = None
            self.version = "unknown"
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(resolved_path), local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(resolved_path))

        self.model.to(self.device)
        self.model.eval()
        self.model_path = str(resolved_path)
        self.version = "v2" if "v2" in resolved_path.name.lower() else "v1"

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        else:
            self.id2label = DEFAULT_ID2LABEL.copy()

        self.model_dir = load_dir

    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        if not self.is_loaded():
            raise RuntimeError("DistilBERT model is not loaded.")

        cleaned = clean_text_for_transformer(text)
        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 0.3333
            }

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            top_idx = int(torch.argmax(probs).item())

        return {
            "sentiment": self.id2label.get(top_idx, "neutral"),
            "confidence": float(probs[top_idx].item())
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        if not self.is_loaded():
            raise RuntimeError("DistilBERT model is not loaded.")

        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            cleaned_chunk = [clean_text_for_transformer(t) for t in chunk]

            non_empty_indices = [idx for idx, c in enumerate(cleaned_chunk) if c]
            batch_results = [
                {"sentiment": "neutral", "confidence": 0.3333} for _ in range(len(chunk))
            ]

            if non_empty_indices:
                texts_to_score = [cleaned_chunk[idx] for idx in non_empty_indices]
                inputs = self.tokenizer(
                    texts_to_score,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    top_indices = torch.argmax(probs, dim=-1)

                for local_pos, orig_idx in enumerate(non_empty_indices):
                    p_idx = int(top_indices[local_pos].item())
                    batch_results[orig_idx] = {
                        "sentiment": self.id2label.get(p_idx, "neutral"),
                        "confidence": float(probs[local_pos][p_idx].item())
                    }

            results.extend(batch_results)

        return results
