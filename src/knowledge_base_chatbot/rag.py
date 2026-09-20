from __future__ import annotations

from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from .config import AppConfig


SYSTEM_PROMPT = """তুমি একটি জ্ঞানভিত্তিক বাংলা বই-চ্যাটবট।

নিয়ম:
1. শুধুমাত্র দেওয়া প্রসঙ্গ থেকে উত্তর দাও।
2. প্রসঙ্গে উত্তর না থাকলে বলবে: "এই তথ্যটি নির্বাচিত বইয়ে পাওয়া যায়নি।"
3. অনুমান, বাইরের জ্ঞান, বা গল্পের সাধারণ পরিচিতি ব্যবহার করবে না।
4. উত্তর সংক্ষিপ্ত কিন্তু পূর্ণাঙ্গ হবে।
5. শেষে "উৎস:" লিখে প্রাসঙ্গিক অধ্যায়/পরিচ্ছেদ দাও।
"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]
    context: list[Document]


def _format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(documents, start=1):
        chapter = doc.metadata.get("chapter", "অজানা অধ্যায়")
        source = doc.metadata.get("source", "")
        blocks.append(f"[{i}] অধ্যায়: {chapter}\nURL: {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _unique_sources(documents: list[Document]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for doc in documents:
        chapter = str(doc.metadata.get("chapter", ""))
        source = str(doc.metadata.get("source", ""))
        key = (chapter, source)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"chapter": chapter, "source": source})
    return sources


def _sources_for_answer(answer: str, documents: list[Document]) -> list[dict[str, str]]:
    if "পাওয়া যায়নি" in answer:
        return []

    cited_docs = [
        doc for doc in documents if str(doc.metadata.get("chapter", "")) in answer
    ]
    if not cited_docs:
        cited_docs = documents[:2]
    return _unique_sources(cited_docs)


def answer_question(question: str, vector_store, config: AppConfig) -> RagAnswer:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.retriever.k},
    )
    docs = retriever.invoke(question)

    if not docs:
        return RagAnswer(
            answer="এই তথ্যটি নির্বাচিত বইয়ে পাওয়া যায়নি।",
            sources=[],
            context=[],
        )

    llm = ChatGoogleGenerativeAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    context = _format_context(docs)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"প্রসঙ্গ:\n{context}\n\n"
        f"প্রশ্ন: {question}\n\n"
        "উত্তর:"
    )
    response = llm.invoke(prompt)
    answer = str(response.content).strip()
    return RagAnswer(answer=answer, sources=_sources_for_answer(answer, docs), context=docs)
