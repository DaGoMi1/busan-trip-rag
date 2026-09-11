from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from data.pipeline.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingModel:
    """한국어 sentence-transformers 임베딩 (+ 쿼리 캐시)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        missing = [text for text in texts if text not in self._cache]
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            vectors = self.model.encode(
                unique_missing,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype="float32")
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for text, vector in zip(unique_missing, vectors, strict=True):
                self._cache[text] = np.asarray(vector, dtype="float32")

        stacked = np.stack([self._cache[text] for text in texts], axis=0)
        return stacked.astype("float32", copy=False)
