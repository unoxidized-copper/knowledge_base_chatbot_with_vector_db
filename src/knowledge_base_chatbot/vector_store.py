from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embeddings import MultilingualE5Embeddings


def build_vector_store(chunks: list[Document], embeddings: MultilingualE5Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No chunks were supplied for indexing.")
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(path: Path, embeddings: MultilingualE5Embeddings) -> FAISS:
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}")
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
