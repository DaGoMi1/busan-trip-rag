from sentence_transformers import SentenceTransformer

from data.pipeline.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingModel:
    """한국어 sentence-transformers 임베딩."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
