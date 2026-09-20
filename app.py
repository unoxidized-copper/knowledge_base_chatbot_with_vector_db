from __future__ import annotations

import streamlit as st

from src.knowledge_base_chatbot.config import load_config
from src.knowledge_base_chatbot.embeddings import MultilingualE5Embeddings
from src.knowledge_base_chatbot.rag import answer_question
from src.knowledge_base_chatbot.vector_store import load_vector_store


st.set_page_config(page_title="Bangla Knowledge Base Chatbot", layout="centered")


@st.cache_resource(show_spinner="Loading vector index...")
def load_resources():
    config = load_config()
    embeddings = MultilingualE5Embeddings(config.embedding.model_name)
    vector_store = load_vector_store(config.paths.vector_store, embeddings)
    return config, vector_store


st.title("Knowledge Base Chatbot")
st.caption("Book: দেবী চৌধুরাণী | Source: Bengali Wikisource")

try:
    config, vector_store = load_resources()
except Exception as exc:
    st.error("Vector store is not ready yet. Run `python scripts/build_index.py` first.")
    st.exception(exc)
    st.stop()

question = st.text_area("Ask a question in Bangla", height=90, placeholder="প্রফুল্ল শ্বশুরবাড়ি যেতে চেয়েছিল কেন?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the book and preparing the answer..."):
        result = answer_question(question.strip(), vector_store, config)

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Sources")
    for source in result.sources:
        st.markdown(f"- **{source['chapter']}**  \n  {source['source']}")

    with st.expander("Retrieved context"):
        for doc in result.context:
            st.markdown(f"**{doc.metadata.get('chapter')}**")
            st.write(doc.page_content[:900] + ("..." if len(doc.page_content) > 900 else ""))
