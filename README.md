# Knowledge Base Chatbot with Vector DB

This project is a Bengali RAG chatbot over one complete prose book from Bengali Wikisource. It crawls chapter subpages, chunks and embeds the text, stores the vectors in FAISS, then answers questions with chapter citations.

## Book Information

- **Book:** দেবী চৌধুরাণী
- **Author:** বঙ্কিমচন্দ্র চট্টোপাধ্যায়
- **Source:** https://bn.wikisource.org/wiki/দেবী_চৌধুরাণী_(বঙ্কিমচন্দ্র_চট্টোপাধ্যায়,_১৯৩৯)
- **Type:** Bengali prose novel
- **Why this book:** The Wikisource edition is available as clean digital text and has separate chapter subpages, so it is suitable for a full-book RAG pipeline without OCR.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project folder:

```env
GEMINI_API_KEY=your_api_key_here
```

`GEMINI_API_KEY` is read automatically and mapped to `GOOGLE_API_KEY` for LangChain's Gemini integration.

## Build the Vector Database

```bash
python scripts/build_index.py
```

This command:

1. Crawls all chapter subpages linked from the Wikisource table of contents.
2. Cleans navigation/footer noise from every chapter.
3. Splits the chapter text into overlapping chunks.
4. Creates multilingual embeddings.
5. Saves a FAISS vector database under `vectorstores/debi_chowdhurani_faiss`.

## Run the Chatbot

```bash
streamlit run app.py
```

You can also ask from the terminal:

```bash
python scripts/ask.py "প্রফুল্ল শ্বশুরবাড়ি যেতে চেয়েছিল কেন?"
```

## Technical Details

### Embedding Model

The project uses `intfloat/multilingual-e5-small`.

I selected it because E5 is trained for multilingual retrieval and supports cross-lingual and non-English semantic search. Bengali questions and Bengali book passages are embedded in the same vector space. The implementation also uses the recommended E5 prefixes:

- `passage:` for book chunks
- `query:` for user questions

This is better than an English-only embedding model because the source text and most questions are in Bangla.

### Chunking

- **Chunk size:** 900 characters
- **Chunk overlap:** 120 characters
- **Separators:** paragraph breaks, line breaks, Bengali danda (`।`), question/exclamation marks, spaces

The chunk size is large enough to preserve a useful scene or explanation, while the overlap keeps important names and references from being split too sharply between chunks.

### Preprocessing

The crawler removes Wikisource navigation tables, edit links, scripts, styles, category/footer text, repeated download/navigation lines, and extra whitespace. Useful metadata is preserved for every document:

- book name
- chapter name
- section name
- source URL
- chunk id

### Vector Database and Retriever

- **Vector database:** FAISS
- **Retriever:** LangChain FAISS retriever
- **Retriever configuration:** top `k=5` similarity search

FAISS is lightweight, local, fast, and easy to rebuild from the crawler output.

### LLM

- **LLM:** Gemini through `langchain-google-genai`
- **Default model:** `gemini-2.5-flash`
- **Temperature:** `0.1`

The prompt tells the model to answer only from the retrieved book context. If the context does not contain the answer, it must say: `এই তথ্যটি নির্বাচিত বইয়ে পাওয়া যায়নি।`

## RAG Pipeline

`Wikisource -> Crawling -> Cleaning -> Chunking -> Embeddings -> FAISS Vector DB -> LangChain Retrieval -> Gemini LLM -> Answer + Citation`

For every question, the system embeds the query, retrieves the most relevant book chunks from FAISS, sends only those chunks to Gemini, and returns an answer with chapter/source citations.

## Test Questions

Ten test questions are in `test_questions.csv`. They include expected answers and source chapters. The final row is a no-answer test to check that the chatbot refuses to hallucinate when the answer is not in the book.

## Bonus: Chunking Comparison

Run:

```bash
python scripts/evaluate_retrieval.py
```

It compares two chunking strategies:

- compact chunks: 700 characters, 100 overlap
- balanced chunks: 900 characters, 120 overlap

The script calculates a simple top-5 hit rate using `test_questions.csv` and writes the result to `reports/bonus_comparison.md`.

## End-to-End Smoke Test

```bash
python scripts/smoke_test.py
```

This rebuilds the index, runs the bonus retrieval comparison, asks one answerable question, and asks one no-answer question through the Gemini RAG pipeline.
