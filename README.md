# PDF Assistant

A Streamlit RAG application that indexes one or more PDFs and lets you chat with their contents.

## Project structure

- `app.py`: Streamlit interface and application flow
- `pdf_service.py`: PDF validation, extraction, metadata, and processing cache
- `rag_service.py`: text splitting, embeddings, FAISS, conversation memory, and RAG chain
- `config.py`: application settings and limits
- `tests/`: pytest coverage for PDF processing, RAG behavior, and the Streamlit smoke test

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Replace the `GEMINI_API_KEY` value in `.env` with your Google AI Studio API key, then run:

```bash
streamlit run app.py
```

Note: Text cannot be extracted from scanned PDFs unless they contain an OCR text layer.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests mock external AI components and do not require a Gemini API key or network access.
