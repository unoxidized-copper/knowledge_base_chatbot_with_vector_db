from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class MultilingualE5Embeddings(Embeddings):
    """Small wrapper so e5 gets the query/passsage prefixes it was trained with."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            inputs,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return vector.tolist()
