from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.knowledge_base_chatbot.config import load_config
from src.knowledge_base_chatbot.embeddings import MultilingualE5Embeddings
from src.knowledge_base_chatbot.processing import chapters_to_documents, save_chunks, split_documents
from src.knowledge_base_chatbot.vector_store import build_vector_store, save_vector_store
from src.knowledge_base_chatbot.wikisource import crawl_book, save_chapters


def main() -> None:
    config = load_config()

    print("Crawling Bengali Wikisource chapter subpages...", flush=True)
    chapters = crawl_book(config.book, cache_path=config.paths.raw_chapters)
    save_chapters(chapters, config.paths.raw_chapters)
    print(f"Saved {len(chapters)} chapters to {config.paths.raw_chapters}", flush=True)

    documents = chapters_to_documents(chapters)
    chunks = split_documents(documents, config.chunking)
    save_chunks(chunks, config.paths.chunks)
    print(f"Created {len(chunks)} chunks", flush=True)

    embeddings = MultilingualE5Embeddings(config.embedding.model_name)
    vector_store = build_vector_store(chunks, embeddings)
    save_vector_store(vector_store, config.paths.vector_store)
    print(f"FAISS vector store saved to {config.paths.vector_store}", flush=True)


if __name__ == "__main__":
    main()
