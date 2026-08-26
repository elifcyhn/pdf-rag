import html
import logging

import streamlit as st
from dotenv import load_dotenv

import pdf_service
import rag_service


logger = logging.getLogger(__name__)


def initialize_session() -> None:
    defaults = {
        "conversation": None,
        "messages": [],
        "document_names": [],
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_documents() -> None:
    st.session_state.conversation = None
    st.session_state.messages = []
    st.session_state.document_names = []
    st.session_state.uploader_key += 1


def clear_chat() -> None:
    rag_service.clear_conversation_memory(st.session_state.conversation)
    st.session_state.messages = []


def render_styles() -> None:
    st.markdown("""
        <style>
        .block-container { max-width: 920px; padding-top: 2rem; padding-bottom: 5rem; }
        [data-testid="stSidebar"] { border-right: 1px solid color-mix(in srgb, currentColor 10%, transparent); }
        [data-testid="stChatMessage"] { border: 1px solid color-mix(in srgb, currentColor 12%, transparent); border-radius: 18px; padding: .55rem .8rem; margin-bottom: .8rem; }
        [data-testid="stChatInput"] { border-radius: 18px; }
        .status-card { padding: .8rem 1rem; border-radius: 14px; background: color-mix(in srgb, #4f7cff 10%, transparent); border: 1px solid color-mix(in srgb, #4f7cff 25%, transparent); margin-bottom: 1rem; }
        .source-chip { display: inline-block; padding: .2rem .55rem; margin: .15rem .25rem .15rem 0; border-radius: 999px; background: color-mix(in srgb, #4f7cff 13%, transparent); font-size: .78rem; }
        .source-heading { margin-top: .75rem; font-size: .78rem; font-weight: 600; opacity: .75; }
        .source-note { margin-top: .25rem; font-size: .72rem; opacity: .62; }
        </style>
    """, unsafe_allow_html=True)


def render_sources(sources: list[str]) -> None:
    if sources:
        chips = "".join(
            f'<span class="source-chip">{html.escape(label)}</span>'
            for label in sources
        )
        st.markdown(
            '<div class="source-heading">Sources</div>'
            f"<div>{chips}</div>"
            '<div class="source-note">Sources identify the PDF pages of the retrieved text chunks; they are not definitive academic citations.</div>',
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    max_files, max_file_size_mb, max_total_pages = pdf_service.get_upload_limits()
    with st.sidebar:
        st.title("📚 Documents")
        st.caption("Upload your PDFs and ask the assistant to use only information from them.")
        st.caption(
            f"Up to {max_files} PDFs · {max_file_size_mb} MB per file · "
            f"{max_total_pages} pages in total"
        )
        files = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"pdf_uploader_{st.session_state.uploader_key}",
        )
        if st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
            disabled=not files,
        ):
            if not rag_service.get_google_api_key():
                st.error("GOOGLE_API_KEY or GEMINI_API_KEY was not found in the `.env` file.")
            else:
                with st.status("Preparing documents...", expanded=True) as document_status:
                    st.write("Validating files and extracting text...")
                    file_payloads = pdf_service.create_file_payloads(files)
                    file_fingerprints = pdf_service.file_fingerprints(file_payloads)
                    pages, rejected_files, processed_names = (
                        pdf_service.process_pdf_payloads(file_payloads)
                    )
                    if rejected_files:
                        st.warning(
                            "Files that could not be processed:\n\n"
                            + "\n\n".join(f"- {message}" for message in rejected_files)
                        )
                    chunks = rag_service.split_documents(pages) if pages else []
                    if not chunks:
                        document_status.update(
                            label="No documents were processed.", state="error"
                        )
                        st.error("No processable text was found. Try a text-based PDF.")
                    else:
                        try:
                            st.write("Creating the search index...")
                            st.session_state.conversation = rag_service.build_conversation(
                                chunks, file_fingerprints
                            )
                            st.session_state.messages = []
                            st.session_state.document_names = processed_names
                            document_status.update(
                                label="Documents are ready.",
                                state="complete",
                                expanded=False,
                            )
                            st.success(
                                f"{len(processed_names)} PDF(s) and "
                                f"{len(chunks)} chunks are ready."
                            )
                        except Exception:
                            logger.exception("Could not index documents")
                            document_status.update(
                                label="Document processing failed.", state="error"
                            )
                            st.error(
                                "The documents could not be prepared. "
                                "Please try again later."
                            )
        if st.session_state.document_names:
            st.divider()
            st.caption("Active documents")
            for name in st.session_state.document_names:
                st.markdown(f"✓ {name}")
        st.divider()
        st.button(
            "Clear documents",
            on_click=clear_documents,
            use_container_width=True,
            disabled=not st.session_state.document_names,
        )
        st.caption("Active documents are attached only to this session.")


def render_message(message: dict) -> None:
    avatar = "👤" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="PDF Assistant",
        page_icon="💬",
        layout="centered",
    )
    initialize_session()
    render_styles()
    render_sidebar()

    st.title("💬 PDF Assistant")
    header_text, header_action = st.columns([5, 1.2], vertical_alignment="center")
    with header_text:
        st.caption(
            "Ask questions, explore ideas, and find answers grounded in your PDF documents."
        )
    with header_action:
        st.button(
            "Clear chat",
            on_click=clear_chat,
            use_container_width=True,
            disabled=not st.session_state.messages,
        )
    if st.session_state.conversation is None:
        st.markdown(
            '<div class="status-card">👈 To begin, upload PDFs from the sidebar '
            'and select <b>Process documents</b>.</div>',
            unsafe_allow_html=True,
        )
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(
                "Hello! Once you upload your documents, I can summarize them, "
                "explain concepts, and compare information across files."
            )

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input(
        "Ask something about your documents...",
        disabled=st.session_state.conversation is None,
    )
    if not prompt:
        return
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Preparing your answer..."):
            try:
                response = st.session_state.conversation.invoke({"question": prompt})
                answer = response.get("answer", "No answer could be generated.")
                sources = rag_service.source_labels(
                    response.get("source_documents", [])
                )
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception:
                logger.exception("Could not generate an answer")
                st.error("The answer could not be generated. Please try again.")


if __name__ == "__main__":
    main()
