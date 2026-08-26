import hashlib
from unittest.mock import patch

import config
import pdf_service


class FakeUpload(pdf_service.InMemoryUpload):
    def getvalue(self) -> bytes:
        return super().getvalue()


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


def make_upload(
    name: str = "document.pdf",
    content: bytes = b"%PDF-test",
    file_type: str = "application/pdf",
) -> FakeUpload:
    return FakeUpload(name, file_type, content)


def test_file_payload_contains_name_and_sha256() -> None:
    upload = make_upload(content=b"%PDF-content")

    payload = pdf_service.create_file_payloads([upload])[0]

    assert payload[0] == "document.pdf"
    assert payload[3] == hashlib.sha256(b"%PDF-content").hexdigest()


def test_same_name_with_different_content_has_different_fingerprint() -> None:
    first = pdf_service.create_file_payloads([make_upload(content=b"%PDF-one")])
    second = pdf_service.create_file_payloads([make_upload(content=b"%PDF-two")])

    assert pdf_service.file_fingerprints(first) != pdf_service.file_fingerprints(second)


def test_validation_rejects_unsupported_and_invalid_files() -> None:
    files = [
        make_upload(name="notes.txt", content=b"text", file_type="text/plain"),
        make_upload(name="fake.pdf", content=b"not-a-pdf"),
    ]

    valid, errors = pdf_service.validate_uploaded_files(files)

    assert valid == []
    assert errors == [
        "notes.txt: Only PDF files are supported.",
        "fake.pdf: This is not a valid PDF file.",
    ]


def test_validation_enforces_file_count_and_size(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_PDF_FILES", 1)
    valid, errors = pdf_service.validate_uploaded_files([make_upload(), make_upload()])
    assert valid == []
    assert errors == ["You can upload up to 1 PDFs at a time."]

    monkeypatch.setattr(config, "MAX_FILE_SIZE_MB", 0)
    valid, errors = pdf_service.validate_uploaded_files([make_upload()])
    assert valid == []
    assert errors == ["document.pdf: The file exceeds the 0 MB limit."]


def test_extracts_text_with_source_and_page_metadata() -> None:
    class Reader:
        is_encrypted = False
        pages = [FakePage("First page"), FakePage("Second page")]

        def __init__(self, uploaded_file) -> None:
            pass

    with patch.object(pdf_service, "PdfReader", Reader):
        documents, errors, names = pdf_service.extract_pdf_pages([make_upload()])

    assert errors == []
    assert names == ["document.pdf"]
    assert [document.page_content for document in documents] == ["First page", "Second page"]
    assert [document.metadata for document in documents] == [
        {"source": "document.pdf", "page": 1},
        {"source": "document.pdf", "page": 2},
    ]


def test_encrypted_textless_and_corrupted_pdfs_return_simple_errors() -> None:
    class EncryptedReader:
        is_encrypted = True
        pages = []

        def __init__(self, uploaded_file) -> None:
            pass

    with patch.object(pdf_service, "PdfReader", EncryptedReader):
        _, errors, _ = pdf_service.extract_pdf_pages([make_upload(name="encrypted.pdf")])
    assert errors == ["encrypted.pdf: Password-protected PDFs are not supported."]

    class TextlessReader:
        is_encrypted = False
        pages = [FakePage(None)]

        def __init__(self, uploaded_file) -> None:
            pass

    with patch.object(pdf_service, "PdfReader", TextlessReader):
        _, errors, _ = pdf_service.extract_pdf_pages([make_upload(name="scan.pdf")])
    assert errors == ["scan.pdf: No readable text was found."]

    with patch.object(pdf_service, "PdfReader", side_effect=ValueError("technical detail")):
        _, errors, _ = pdf_service.extract_pdf_pages([make_upload(name="broken.pdf")])
    assert errors == ["broken.pdf: The PDF could not be read or is corrupted."]
    assert "technical detail" not in errors[0]


def test_total_page_limit_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_TOTAL_PAGES", 1)

    class Reader:
        is_encrypted = False
        pages = [FakePage("One"), FakePage("Two")]

        def __init__(self, uploaded_file) -> None:
            pass

    with patch.object(pdf_service, "PdfReader", Reader):
        documents, errors, names = pdf_service.extract_pdf_pages([make_upload()])

    assert documents == []
    assert names == []
    assert errors == [
        "document.pdf: Not processed because the total limit of 1 pages would be exceeded."
    ]


def test_same_payload_uses_cached_pdf_extraction() -> None:
    calls = 0

    class Reader:
        is_encrypted = False
        pages = [FakePage("Cached text")]

        def __init__(self, uploaded_file) -> None:
            nonlocal calls
            calls += 1

    payloads = pdf_service.create_file_payloads([make_upload(content=b"%PDF-cache")])
    pdf_service.process_pdf_payloads.clear()
    try:
        with patch.object(pdf_service, "PdfReader", Reader):
            first = pdf_service.process_pdf_payloads(payloads)
            second = pdf_service.process_pdf_payloads(payloads)
        assert first == second
        assert calls == 1
    finally:
        pdf_service.process_pdf_payloads.clear()
