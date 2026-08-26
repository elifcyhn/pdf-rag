import html
import hashlib
import logging
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


MAX_PDF_FILES = 5
MAX_FILE_SIZE_MB = 20
MAX_TOTAL_PAGES = 500
CHAT_MEMORY_TURNS = 6
CACHE_TTL_SECONDS = 3600
CACHE_MAX_ENTRIES = 32
EMBEDDING_MODEL = "models/gemini-embedding-2"
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}

logger = logging.getLogger(__name__)


ANSWER_PROMPT = PromptTemplate.from_template("""
Answer the question using only information explicitly found in the PDF context below.
If the context does not contain the answer, do not guess, infer, or add general knowledge.
In that case, write exactly this sentence and nothing else: This information was not found in the uploaded documents.
Answer in English.

Context:
{context}

Question: {question}
Answer:
""")


def initialize_session() -> None:
    for key, value in {"conversation": None, "messages": [], "document_names": []}.items():
        if key not in st.session_state:
            st.session_state[key] = value


class InMemoryUpload(BytesIO):
    def __init__(self, name: str, file_type: str | None, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.type = file_type
        self.size = len(content)


def create_file_payloads(files) -> tuple[tuple[str, str | None, bytes, str], ...]:
    payloads = []
    for uploaded_file in files:
        content = uploaded_file.getvalue()
        payloads.append((
            uploaded_file.name,
            getattr(uploaded_file, "type", None),
            content,
            hashlib.sha256(content).hexdigest(),
        ))
    return tuple(payloads)


def validate_uploaded_files(files) -> tuple[list, list[str]]:
    valid_files, errors = [], []
    if len(files) > MAX_PDF_FILES:
        return [], [f"You can upload up to {MAX_PDF_FILES} PDFs at a time."]

    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    for uploaded_file in files:
        filename = getattr(uploaded_file, "name", "Unnamed file")
        file_type = getattr(uploaded_file, "type", None)
        if Path(filename).suffix.lower() != ".pdf" or (
            file_type and file_type not in PDF_MIME_TYPES
        ):
            errors.append(f"{filename}: Only PDF files are supported.")
            continue
        if getattr(uploaded_file, "size", 0) > max_size_bytes:
            errors.append(f"{filename}: The file exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            continue
        try:
            uploaded_file.seek(0)
            signature = uploaded_file.read(5)
            uploaded_file.seek(0)
        except Exception:
            logger.exception("Could not read the uploaded file header: %s", filename)
            errors.append(f"{filename}: The file could not be read.")
            continue
        if signature != b"%PDF-":
            errors.append(f"{filename}: This is not a valid PDF file.")
            continue
        valid_files.append(uploaded_file)
    return valid_files, errors


def extract_pdf_pages(files) -> tuple[list[Document], list[str], list[str]]:
    documents, errors, processed_names = [], [], []
    total_pages = 0
    for uploaded_file in files:
        try:
            uploaded_file.seek(0)
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                errors.append(f"{uploaded_file.name}: Password-protected PDFs are not supported.")
                continue
            pdf_page_count = len(reader.pages)
            if total_pages + pdf_page_count > MAX_TOTAL_PAGES:
                errors.append(
                    f"{uploaded_file.name}: Not processed because the total limit of {MAX_TOTAL_PAGES} pages would be exceeded."
                )
                continue
            total_pages += pdf_page_count
            file_documents = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    file_documents.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": page_number},
                    ))
            if not file_documents:
                errors.append(f"{uploaded_file.name}: No readable text was found.")
                continue
            documents.extend(file_documents)
            processed_names.append(uploaded_file.name)
        except Exception:
            logger.exception("Could not process PDF: %s", uploaded_file.name)
            errors.append(f"{uploaded_file.name}: The PDF could not be read or is corrupted.")
    return documents, errors, processed_names


def split_documents(documents: list[Document]) -> list[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(documents)


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def process_pdf_payloads(
    payloads: tuple[tuple[str, str | None, bytes, str], ...],
) -> tuple[list[Document], list[str], list[str]]:
    files = []
    for name, file_type, content, content_hash in payloads:
        if hashlib.sha256(content).hexdigest() != content_hash:
            logger.error("PDF cache fingerprint mismatch: %s", name)
            return [], [f"{name}: The file could not be verified."], []
        files.append(InMemoryUpload(name, file_type, content))

    valid_files, validation_errors = validate_uploaded_files(files)
    pages, processing_errors, processed_names = extract_pdf_pages(valid_files)
    chunks = split_documents(pages) if pages else []
    return chunks, validation_errors + processing_errors, processed_names


def get_google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def create_document_payloads(
    documents: list[Document],
) -> tuple[tuple[str, tuple[tuple[str, str | int], ...]], ...]:
    return tuple(
        (document.page_content, tuple(sorted(document.metadata.items())))
        for document in documents
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def get_cached_document_embeddings(
    document_payloads: tuple[tuple[str, tuple[tuple[str, str | int], ...]], ...],
    file_fingerprints: tuple[tuple[str, str], ...],
) -> list[list[float]]:
    del file_fingerprints  # Included only to bind the cache entry to names and content hashes.
    embedding_client = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_google_api_key(),
    )
    return embedding_client.embed_documents([text for text, _ in document_payloads])


def build_conversation(
    documents: list[Document], file_fingerprints: tuple[tuple[str, str], ...]
):
    api_key = get_google_api_key()
    embedding_client = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    document_payloads = create_document_payloads(documents)
    vectors = get_cached_document_embeddings(document_payloads, file_fingerprints)
    vectorstore = FAISS.from_embeddings(
        text_embeddings=(
            (text, vector)
            for (text, _), vector in zip(document_payloads, vectors, strict=True)
        ),
        embedding=embedding_client,
        metadatas=(dict(metadata) for _, metadata in document_payloads),
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=CHAT_MEMORY_TURNS,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model="gemini-3.6-flash", temperature=0.2, google_api_key=api_key
        ),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
    )


def source_labels(source_documents: list[Document]) -> list[str]:
    pages_by_file = defaultdict(set)
    for document in source_documents:
        source = document.metadata.get("source", "Document")
        page = document.metadata.get("page")
        if isinstance(page, int):
            pages_by_file[source].add(page)
    return [
        f"{source} · p. {', '.join(map(str, sorted(pages)))}" if pages else source
        for source, pages in pages_by_file.items()
    ]


def reset_chat() -> None:
    st.session_state.conversation = None
    st.session_state.messages = []
    st.session_state.document_names = []


def render_styles() -> None:
    st.markdown("""
        <style>
        .block-container { max-width: 920px; padding-top: 2.2rem; }
        [data-testid="stChatMessage"] { border: 1px solid color-mix(in srgb, currentColor 12%, transparent); border-radius: 18px; padding: .45rem .75rem; margin-bottom: .75rem; }
        [data-testid="stChatInput"] { border-radius: 18px; }
        .status-card { padding: .8rem 1rem; border-radius: 14px; background: color-mix(in srgb, #4f7cff 10%, transparent); border: 1px solid color-mix(in srgb, #4f7cff 25%, transparent); margin-bottom: 1rem; }
        .source-chip { display: inline-block; padding: .2rem .55rem; margin: .15rem .25rem .15rem 0; border-radius: 999px; background: color-mix(in srgb, #4f7cff 13%, transparent); font-size: .78rem; }
        </style>
    """, unsafe_allow_html=True)


def render_sources(sources: list[str]) -> None:
    if sources:
        chips = "".join(f'<span class="source-chip">{html.escape(label)}</span>' for label in sources)
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("📚 Documents")
        st.caption("Upload your PDFs and ask the assistant to use only information from them.")
        st.caption(
            f"Up to {MAX_PDF_FILES} PDFs · {MAX_FILE_SIZE_MB} MB per file · "
            f"{MAX_TOTAL_PAGES} pages in total"
        )
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Process documents", type="primary", use_container_width=True, disabled=not files):
            if not get_google_api_key():
                st.error("GOOGLE_API_KEY or GEMINI_API_KEY was not found in the `.env` file.")
            else:
                with st.spinner("Reading and indexing documents..."):
                    file_payloads = create_file_payloads(files)
                    file_fingerprints = tuple(
                        (name, content_hash)
                        for name, _, _, content_hash in file_payloads
                    )
                    chunks, rejected_files, processed_names = process_pdf_payloads(
                        file_payloads
                    )
                    if rejected_files:
                        st.warning("Files that could not be processed:\n\n" + "\n\n".join(
                            f"- {message}" for message in rejected_files
                        ))
                    chunks = split_documents(pages) if pages else []
                    if not chunks:
                        st.error("No processable text was found. Try a text-based PDF.")
                    else:
                        try:
                            st.session_state.conversation = build_conversation(
                                chunks, file_fingerprints
                            )
                            st.session_state.messages = []
                            st.session_state.document_names = processed_names
                            st.success(f"{len(processed_names)} PDF(s) and {len(chunks)} chunks are ready.")
                        except Exception:
                            logger.exception("Could not index documents")
                            st.error("The documents could not be prepared. Please try again later.")
        if st.session_state.document_names:
            st.divider()
            st.caption("Active documents")
            for name in st.session_state.document_names:
                st.markdown(f"✓ {name}")
        st.divider()
        st.button("Clear chat and documents", on_click=reset_chat, use_container_width=True)
        st.caption("Files are kept in memory only for this session.")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="PDF Assistant", page_icon="💬", layout="centered")
    initialize_session()
    render_styles()
    render_sidebar()

    st.title("💬 PDF Assistant")
    st.caption("Chat naturally with your documents and see which pages support each answer.")
    st.caption("Source pages are a helpful indicator, not a definitive academic citation.")
    if st.session_state.conversation is None:
        st.markdown('<div class="status-card">👈 To begin, upload PDFs from the sidebar and select <b>Process documents</b>.</div>', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("Hello! Once you upload your documents, I can summarize them, explain concepts, and compare information across files.")

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("Ask something about your documents...", disabled=st.session_state.conversation is None)
    if not prompt:
        return
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Searching the documents..."):
            try:
                response = st.session_state.conversation.invoke({"question": prompt})
                answer = response.get("answer", "No answer could be generated.")
                sources = source_labels(response.get("source_documents", []))
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception:
                logger.exception("Could not generate an answer")
                st.error("The answer could not be generated. Please try again.")


if __name__ == "__main__":
    main()
