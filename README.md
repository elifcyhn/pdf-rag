# PDF Assistant

A Streamlit RAG application that indexes one or more PDFs and lets you chat with their contents.

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
