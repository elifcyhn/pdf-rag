from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

import config
import rag_service


def test_split_documents_preserves_metadata() -> None:
    documents = [
        Document(
            page_content="A sentence. " * 300,
            metadata={"source": "document.pdf", "page": 4},
        )
    ]

    chunks = rag_service.split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.metadata == {"source": "document.pdf", "page": 4} for chunk in chunks)


def test_prompt_preserves_grounding_and_question_language_rules() -> None:
    prompt = rag_service.ANSWER_PROMPT.template

    assert "do not guess, infer, or add general knowledge" in prompt
    assert "Detect the primary language of the question" in prompt
    assert "Bu bilgi yüklenen belgelerde bulunamadı." in prompt
    assert "This information was not found in the uploaded documents." in prompt


def test_document_payload_preserves_text_and_metadata() -> None:
    document = Document(
        page_content="Text",
        metadata={"page": 2, "source": "document.pdf"},
    )

    payloads = rag_service.create_document_payloads([document])

    assert payloads == (("Text", (("page", 2), ("source", "document.pdf"))),)


def test_embedding_cache_avoids_duplicate_api_calls() -> None:
    calls = 0

    class Embeddings:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model"] == config.EMBEDDING_MODEL

        def embed_documents(self, texts):
            nonlocal calls
            calls += 1
            return [[1.0, 0.0] for _ in texts]

    payloads = (("Text", (("page", 1), ("source", "document.pdf"))),)
    fingerprints = (("document.pdf", "sha256"),)
    rag_service.get_cached_document_embeddings.clear()
    try:
        with patch.object(rag_service, "GoogleGenerativeAIEmbeddings", Embeddings):
            first = rag_service.get_cached_document_embeddings(payloads, fingerprints)
            second = rag_service.get_cached_document_embeddings(payloads, fingerprints)
        assert first == second
        assert calls == 1
    finally:
        rag_service.get_cached_document_embeddings.clear()


def test_build_conversation_keeps_six_turn_window_without_network() -> None:
    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            return ("retriever", kwargs)

    chain = Mock()
    with (
        patch.object(rag_service, "GoogleGenerativeAIEmbeddings", return_value=Mock()),
        patch.object(rag_service, "ChatGoogleGenerativeAI", return_value=Mock()),
        patch.object(rag_service, "get_cached_document_embeddings", return_value=[[1.0, 0.0]]),
        patch.object(rag_service.FAISS, "from_embeddings", return_value=FakeVectorstore()),
        patch.object(
            rag_service.ConversationalRetrievalChain,
            "from_llm",
            return_value=chain,
        ) as from_llm,
    ):
        result = rag_service.build_conversation(
            [Document(page_content="Text", metadata={"source": "doc.pdf", "page": 1})],
            (("doc.pdf", "hash"),),
        )

    assert result is chain
    memory = from_llm.call_args.kwargs["memory"]
    assert isinstance(memory, rag_service.ConversationBufferWindowMemory)
    assert memory.k == config.CHAT_MEMORY_TURNS == 6
    assert from_llm.call_args.kwargs["retriever"] == (
        "retriever",
        {"search_kwargs": {"k": config.RETRIEVER_DOCUMENT_COUNT}},
    )


def test_source_labels_group_and_sort_pages() -> None:
    documents = [
        Document(page_content="", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="", metadata={"source": "b.pdf", "page": 2}),
    ]

    assert rag_service.source_labels(documents) == ["a.pdf · p. 1, 3", "b.pdf · p. 2"]


def test_clear_conversation_memory() -> None:
    memory = Mock()
    conversation = SimpleNamespace(memory=memory)

    rag_service.clear_conversation_memory(conversation)

    memory.clear.assert_called_once_with()
