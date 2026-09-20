from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.knowledge_base_chatbot.config import load_config
from src.knowledge_base_chatbot.embeddings import MultilingualE5Embeddings
from src.knowledge_base_chatbot.rag import answer_question
from src.knowledge_base_chatbot.vector_store import load_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG chatbot from the terminal.")
    parser.add_argument("question", help="Question to ask in Bangla")
    args = parser.parse_args()

    config = load_config()
    embeddings = MultilingualE5Embeddings(config.embedding.model_name)
    vector_store = load_vector_store(config.paths.vector_store, embeddings)
    result = answer_question(args.question, vector_store, config)

    print("\nAnswer\n------")
    print(result.answer)
    print("\nRetrieved sources\n-----------------")
    for source in result.sources:
        print(f"- {source['chapter']}: {source['source']}")


if __name__ == "__main__":
    main()
