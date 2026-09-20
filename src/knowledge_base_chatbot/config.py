from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BookConfig:
    title: str
    author: str
    root_url: str
    description: str
    chapter_link_must_contain: str = ""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str


@dataclass(frozen=True)
class RetrieverConfig:
    k: int


@dataclass(frozen=True)
class LlmConfig:
    model: str
    temperature: float


@dataclass(frozen=True)
class PathConfig:
    raw_chapters: Path
    chunks: Path
    vector_store: Path
    bonus_report: Path


@dataclass(frozen=True)
class AppConfig:
    book: BookConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    llm: LlmConfig
    paths: PathConfig


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path | None = None) -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")

    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    config_file = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with config_file.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file)

    book = data["book"]
    paths = data["paths"]
    chunking = data["chunking"]
    embedding = data["embedding"]
    retriever = data["retriever"]
    llm = data["llm"]

    return AppConfig(
        book=BookConfig(**book),
        chunking=ChunkingConfig(**chunking),
        embedding=EmbeddingConfig(**embedding),
        retriever=RetrieverConfig(**retriever),
        llm=LlmConfig(**llm),
        paths=PathConfig(
            raw_chapters=_project_path(paths["raw_chapters"]),
            chunks=_project_path(paths["chunks"]),
            vector_store=_project_path(paths["vector_store"]),
            bonus_report=_project_path(paths["bonus_report"]),
        ),
    )
