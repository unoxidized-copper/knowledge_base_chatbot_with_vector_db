from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ChunkingConfig
from .wikisource import Chapter


def preprocess_text(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chapters_to_documents(chapters: list[Chapter]) -> list[Document]:
    documents: list[Document] = []
    for chapter in chapters:
        text = preprocess_text(chapter.text)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "book": chapter.book_title,
                    "chapter": chapter.chapter,
                    "section": chapter.section,
                    "source": chapter.source_url,
                },
            )
        )
    return documents


def split_documents(documents: list[Document], chunking: ChunkingConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        separators=["\n\n", "\n", "।", "?", "!", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = chunking.chunk_size
        chunk.metadata["chunk_overlap"] = chunking.chunk_overlap
    return chunks


def save_chunks(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunks]
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def load_chunks(path: Path) -> list[Document]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return [Document(page_content=row["page_content"], metadata=row["metadata"]) for row in rows]
