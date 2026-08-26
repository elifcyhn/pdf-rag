import ast
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyPDF2 import PdfWriter
from streamlit.testing.v1 import AppTest

import app


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_only_imports_expected_local_services() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "pdf_service" in imported_modules
    assert "rag_service" in imported_modules
    assert "config" not in imported_modules


def test_clear_chat_preserves_documents_and_index() -> None:
    conversation = object()
    state = SimpleNamespace(
        conversation=conversation,
        messages=[{"role": "user", "content": "Question"}],
        document_names=["document.pdf"],
        uploader_key=0,
    )
    with (
        patch.object(app.st, "session_state", state),
        patch.object(app.rag_service, "clear_conversation_memory") as clear_memory,
    ):
        app.clear_chat()

    clear_memory.assert_called_once_with(conversation)
    assert state.conversation is conversation
    assert state.document_names == ["document.pdf"]
    assert state.messages == []


def test_clear_documents_resets_only_active_session() -> None:
    state = SimpleNamespace(
        conversation=object(),
        messages=[{"role": "user", "content": "Question"}],
        document_names=["document.pdf"],
        uploader_key=2,
    )
    with patch.object(app.st, "session_state", state):
        app.clear_documents()

    assert state.conversation is None
    assert state.messages == []
    assert state.document_names == []
    assert state.uploader_key == 3


def test_streamlit_smoke_without_api_or_network() -> None:
    at = AppTest.from_file(APP_PATH).run(timeout=20)

    assert not at.exception
    assert "💬 PDF Assistant" in [item.value for item in at.title]
    assert {"Process documents", "Clear documents", "Clear chat"}.issubset(
        {button.label for button in at.button}
    )


def test_empty_pdf_upload_path_does_not_call_gemini() -> None:
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)

    previous_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "test-key-not-used"
    try:
        at = AppTest.from_file(APP_PATH).run(timeout=20)
        at.file_uploader[0].upload(
            "empty.pdf", buffer.getvalue(), "application/pdf"
        ).run(timeout=20)
        process_button = next(
            button for button in at.button if button.label == "Process documents"
        )
        process_button.click().run(timeout=20)
    finally:
        if previous_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = previous_key

    assert not at.exception
    assert any("No processable text was found" in error.value for error in at.error)
