import hashlib
import logging
from io import BytesIO
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from PyPDF2 import PdfReader

import config


logger = logging.getLogger(__name__)

FilePayload = tuple[str, str | None, bytes, str]


class InMemoryUpload(BytesIO):
    def __init__(self, name: str, file_type: str | None, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.type = file_type
        self.size = len(content)


def get_upload_limits() -> tuple[int, int, int]:
    return (
        config.MAX_PDF_FILES,
        config.MAX_FILE_SIZE_MB,
        config.MAX_TOTAL_PAGES,
    )


def create_file_payloads(files) -> tuple[FilePayload, ...]:
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


def file_fingerprints(payloads: tuple[FilePayload, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((name, content_hash) for name, _, _, content_hash in payloads)


def validate_uploaded_files(files) -> tuple[list, list[str]]:
    valid_files, errors = [], []
    if len(files) > config.MAX_PDF_FILES:
        return [], [f"You can upload up to {config.MAX_PDF_FILES} PDFs at a time."]

    max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    for uploaded_file in files:
        filename = getattr(uploaded_file, "name", "Unnamed file")
        file_type = getattr(uploaded_file, "type", None)
        if Path(filename).suffix.lower() != ".pdf" or (
            file_type and file_type not in config.PDF_MIME_TYPES
        ):
            errors.append(f"{filename}: Only PDF files are supported.")
            continue
        if getattr(uploaded_file, "size", 0) > max_size_bytes:
            errors.append(f"{filename}: The file exceeds the {config.MAX_FILE_SIZE_MB} MB limit.")
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
            if total_pages + pdf_page_count > config.MAX_TOTAL_PAGES:
                errors.append(
                    f"{uploaded_file.name}: Not processed because the total limit of "
                    f"{config.MAX_TOTAL_PAGES} pages would be exceeded."
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


@st.cache_data(
    ttl=config.CACHE_TTL_SECONDS,
    max_entries=config.CACHE_MAX_ENTRIES,
    show_spinner=False,
)
def process_pdf_payloads(
    payloads: tuple[FilePayload, ...],
) -> tuple[list[Document], list[str], list[str]]:
    files = []
    for name, file_type, content, content_hash in payloads:
        if hashlib.sha256(content).hexdigest() != content_hash:
            logger.error("PDF cache fingerprint mismatch: %s", name)
            return [], [f"{name}: The file could not be verified."], []
        files.append(InMemoryUpload(name, file_type, content))

    valid_files, validation_errors = validate_uploaded_files(files)
    pages, processing_errors, processed_names = extract_pdf_pages(valid_files)
    return pages, validation_errors + processing_errors, processed_names
