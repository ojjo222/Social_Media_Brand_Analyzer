from __future__ import annotations

from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  
    SentenceTransformer = None  #type


class SentenceEmbeddingGenerator:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: "SentenceTransformer | None" = None

    @property
    def model(self):
        
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers package is not available. "
                    "Install it via 'pip install sentence-transformers'"
                )
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _clean_inputs(self, sentences: List[str]) -> List[str]:
        cleaned = []
        for s in sentences:
            if s is None:
                cleaned.append("")
            elif isinstance(s, str):
                cleaned.append(s.strip() if s.strip() else "")
            else:
                cleaned.append(str(s).strip())
        return cleaned

    def encode(self, sentences: List[str]) -> np.ndarray:
        if not isinstance(sentences, list):
            raise TypeError("`sentences` must be a list of strings")

        cleaned = self._clean_inputs(sentences)
        if not cleaned:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim))

        embeddings = self.model.encode(cleaned, batch_size=64, normalize_embeddings=True)
        return np.asarray(embeddings)

    def __call__(self, sentences: List[str]) -> np.ndarray: 
        return self.encode(sentences)

embedding_generator = SentenceEmbeddingGenerator()

__all__ = ["SentenceEmbeddingGenerator", "embedding_generator"]
