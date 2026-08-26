import os
from collections import defaultdict

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


ANSWER_PROMPT = PromptTemplate.from_template("""
Answer the question using only information explicitly found in the PDF context below.
If the context does not contain the answer, do not guess, infer, or add general knowledge.
Detect the primary language of the question and answer in that same language.
If the answer is absent from the context, respond only with the natural equivalent of
"This information was not found in the uploaded documents." in the question's language.
For Turkish, write exactly: Bu bilgi yüklenen belgelerde bulunamadı.
For English, write exactly: This information was not found in the uploaded documents.

Context:
{context}

Question: {question}
Answer:
""")

DocumentPayload = tuple[str, tuple[tuple[str, str | int], ...]]
FileFingerprint = tuple[str, str]


def split_documents(documents: list[Document]) -> list[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    ).split_documents(documents)


def get_google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def create_document_payloads(
    documents: list[Document],
) -> tuple[DocumentPayload, ...]:
    return tuple(
        (document.page_content, tuple(sorted(document.metadata.items())))
        for document in documents
    )


@st.cache_data(
    ttl=config.CACHE_TTL_SECONDS,
    max_entries=config.CACHE_MAX_ENTRIES,
    show_spinner=False,
)
def get_cached_document_embeddings(
    document_payloads: tuple[DocumentPayload, ...],
    file_fingerprints: tuple[FileFingerprint, ...],
) -> list[list[float]]:
    del file_fingerprints  # Binds the cache entry to file names and content hashes.
    embedding_client = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=get_google_api_key(),
    )
    return embedding_client.embed_documents([text for text, _ in document_payloads])


def build_conversation(
    documents: list[Document],
    file_fingerprints: tuple[FileFingerprint, ...],
):
    api_key = get_google_api_key()
    embedding_client = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
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
        k=config.CHAT_MEMORY_TURNS,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            temperature=config.CHAT_TEMPERATURE,
            google_api_key=api_key,
        ),
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_DOCUMENT_COUNT}
        ),
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


def clear_conversation_memory(conversation) -> None:
    if conversation is not None and conversation.memory is not None:
        conversation.memory.clear()
