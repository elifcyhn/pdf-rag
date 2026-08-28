import hashlib
import logging
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.documents import Document
from PyPDF2 import PdfReader

import config


logger = logging.getLogger(__name__)

FilePayload = tuple[str, str | None, bytes, str]
OcrRuntimeSignature = tuple[str, str, tuple[str, ...]]

OCR_INSTALL_MESSAGE = (
    "Local OCR is unavailable. On macOS, run `brew install ocrmypdf`. "
    "For Turkish support, also install `brew install tesseract-lang`."
)


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


def get_ocr_options() -> tuple[bool, tuple[str, ...], str]:
    return (
        config.OCR_ENABLED_DEFAULT,
        tuple(config.OCR_LANGUAGES),
        config.OCR_DEFAULT_LANGUAGE,
    )


def get_ocr_language_code(label: str) -> str:
    return config.OCR_LANGUAGES[label]


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


def get_ocr_runtime_signature() -> OcrRuntimeSignature:
    ocrmypdf_path = shutil.which("ocrmypdf") or ""
    tesseract_path = shutil.which("tesseract") or ""
    languages: tuple[str, ...] = ()
    if tesseract_path:
        try:
            result = subprocess.run(
                [tesseract_path, "--list-langs"],
                shell=False,
                timeout=10,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                languages = tuple(sorted(
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip() and not line.startswith("List of available languages")
                ))
            else:
                logger.error("Could not list Tesseract languages: %s", result.stderr)
        except (OSError, subprocess.SubprocessError):
            logger.exception("Could not inspect the local Tesseract installation")
    return ocrmypdf_path, tesseract_path, languages


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


def extract_text_documents(reader, source: str) -> list[Document]:
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": source, "page": page_number},
            ))
    return documents


def run_local_ocr(
    content: bytes,
    language: str,
    runtime_signature: OcrRuntimeSignature,
) -> tuple[bytes | None, str | None]:
    if language not in config.OCR_LANGUAGES.values():
        return None, "An invalid OCR language was selected."

    ocrmypdf_path, tesseract_path, available_languages = runtime_signature
    if not ocrmypdf_path or not tesseract_path:
        return None, OCR_INSTALL_MESSAGE

    requested_languages = set(language.split("+"))
    if not requested_languages.issubset(available_languages):
        return None, (
            "The selected OCR language is not installed. For Turkish support, "
            "run `brew install tesseract-lang`."
        )

    try:
        with TemporaryDirectory(prefix="pdf-rag-ocr-") as temp_directory:
            input_path = Path(temp_directory) / "input.pdf"
            output_path = Path(temp_directory) / "output.pdf"
            input_path.write_bytes(content)
            result = subprocess.run(
                [
                    ocrmypdf_path,
                    "--language",
                    language,
                    "--skip-text",
                    "--output-type",
                    "pdf",
                    "--quiet",
                    str(input_path),
                    str(output_path),
                ],
                shell=False,
                timeout=config.OCR_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error("OCRmyPDF failed: %s", result.stderr)
                return None, "OCR could not be completed. Please check the PDF file."
            if not output_path.is_file():
                logger.error("OCRmyPDF completed without creating an output file")
                return None, "OCR could not be completed. Please check the PDF file."
            return output_path.read_bytes(), None
    except subprocess.TimeoutExpired:
        logger.exception("OCRmyPDF timed out")
        return None, "OCR timed out. Try a smaller PDF."
    except OSError:
        logger.exception("Could not run OCRmyPDF")
        return None, OCR_INSTALL_MESSAGE


def extract_pdf_pages(
    files,
    enable_ocr: bool = False,
    ocr_language: str = "eng+tur",
    ocr_runtime_signature: OcrRuntimeSignature = ("", "", ()),
) -> tuple[list[Document], list[str], list[str]]:
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
            file_documents = extract_text_documents(reader, uploaded_file.name)
            if not file_documents:
                if not enable_ocr:
                    errors.append(f"{uploaded_file.name}: No readable text was found.")
                    continue
                ocr_content, ocr_error = run_local_ocr(
                    uploaded_file.getvalue(),
                    ocr_language,
                    ocr_runtime_signature,
                )
                if ocr_error:
                    errors.append(f"{uploaded_file.name}: {ocr_error}")
                    continue
                try:
                    ocr_reader = PdfReader(BytesIO(ocr_content))
                except Exception:
                    logger.exception("Could not read the OCR output: %s", uploaded_file.name)
                    errors.append(
                        f"{uploaded_file.name}: The PDF text could not be read after OCR."
                    )
                    continue
                file_documents = extract_text_documents(
                    ocr_reader, uploaded_file.name
                )
                if not file_documents:
                    errors.append(
                        f"{uploaded_file.name}: No readable text was found after OCR."
                    )
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
    enable_ocr: bool = False,
    ocr_language: str = "eng+tur",
    ocr_runtime_signature: OcrRuntimeSignature = ("", "", ()),
) -> tuple[list[Document], list[str], list[str]]:
    files = []
    for name, file_type, content, content_hash in payloads:
        if hashlib.sha256(content).hexdigest() != content_hash:
            logger.error("PDF cache fingerprint mismatch: %s", name)
            return [], [f"{name}: The file could not be verified."], []
        files.append(InMemoryUpload(name, file_type, content))

    valid_files, validation_errors = validate_uploaded_files(files)
    pages, processing_errors, processed_names = extract_pdf_pages(
        valid_files,
        enable_ocr=enable_ocr,
        ocr_language=ocr_language,
        ocr_runtime_signature=ocr_runtime_signature,
    )
    return pages, validation_errors + processing_errors, processed_names
