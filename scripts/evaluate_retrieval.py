from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.knowledge_base_chatbot.config import ChunkingConfig, load_config
from src.knowledge_base_chatbot.embeddings import MultilingualE5Embeddings
from src.knowledge_base_chatbot.processing import chapters_to_documents, split_documents
from src.knowledge_base_chatbot.vector_store import build_vector_store
from src.knowledge_base_chatbot.wikisource import crawl_book, load_chapters, save_chapters


@dataclass
class StrategyResult:
    name: str
    chunk_size: int
    chunk_overlap: int
    hits: int
    total: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def load_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def is_hit(row: dict[str, str], docs) -> bool:
    expected_source = row["source_chapter"].strip()
    if expected_source.upper() == "N/A":
        return False
    if any(expected_source in doc.metadata.get("chapter", "") for doc in docs):
        return True

    keywords = [part.strip() for part in row.get("expected_keywords", "").split("|") if part.strip()]
    joined_text = "\n".join(doc.page_content for doc in docs)
    return bool(keywords and any(keyword in joined_text for keyword in keywords))


def evaluate_strategy(name: str, chunking: ChunkingConfig, questions: list[dict[str, str]]) -> StrategyResult:
    config = load_config()
    if config.paths.raw_chapters.exists():
        chapters = load_chapters(config.paths.raw_chapters)
    else:
        chapters = crawl_book(config.book)
        save_chapters(chapters, config.paths.raw_chapters)

    chunks = split_documents(chapters_to_documents(chapters), chunking)
    embeddings = MultilingualE5Embeddings(config.embedding.model_name)
    store = build_vector_store(chunks, embeddings)

    hits = 0
    total = 0
    for row in questions:
        if row["source_chapter"].strip().upper() == "N/A":
            continue
        total += 1
        docs = store.similarity_search(row["question"], k=5)
        if is_hit(row, docs):
            hits += 1

    return StrategyResult(name, chunking.chunk_size, chunking.chunk_overlap, hits, total)


def write_report(results: list[StrategyResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda result: result.hit_rate)

    lines = [
        "# Bonus Retrieval Comparison",
        "",
        "The same multilingual embedding model was used for both approaches. Only the chunking settings changed.",
        "",
        "| Approach | Chunk Size | Overlap | Hit Rate | Hits |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.chunk_size} | {result.chunk_overlap} | "
            f"{result.hit_rate:.0%} | {result.hits}/{result.total} |"
        )
    lines.extend(
        [
            "",
            f"Best result: **{best.name}**.",
            "",
            "A hit means that the expected chapter appeared in the top-5 retrieved passages, "
            "or that one of the expected answer keywords appeared in those passages. "
            "The no-answer question is excluded from the hit-rate calculation because it has no correct book passage.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    questions = load_questions(ROOT / "test_questions.csv")
    strategies = [
        ("A: compact chunks", ChunkingConfig(chunk_size=700, chunk_overlap=100)),
        ("B: balanced chunks", ChunkingConfig(chunk_size=900, chunk_overlap=120)),
    ]
    results = [evaluate_strategy(name, chunking, questions) for name, chunking in strategies]
    write_report(results, config.paths.bonus_report)

    for result in results:
        print(f"{result.name}: {result.hit_rate:.0%} ({result.hits}/{result.total})")
    print(f"Report written to {config.paths.bonus_report}")


if __name__ == "__main__":
    main()
