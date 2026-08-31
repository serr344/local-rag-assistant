from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag import LocalRAG


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered",
)


# ---------------------------------------------------------
# Compact sidebar styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        padding: 0.55rem;
        min-height: 80px;
    }

    [data-testid="stSidebar"] hr {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }

    [data-testid="stSidebar"] {
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_documents() -> list[str]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        path.name
        for path in DOCS_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
    )


def ensure_rag() -> LocalRAG:
    """
    Create the RAG instance only once per Streamlit session.
    """
    if "rag" not in st.session_state:
        rag = LocalRAG()
        rag.start()
        st.session_state.rag = rag

    return st.session_state.rag


def show_sources(sources) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            st.markdown(
                f"**{source.source}**  \n"
                f"Chunk: `{source.chunk_index}` · "
                f"Similarity: `{source.score:.3f}`"
            )


def set_notice(
    message: str,
    kind: str = "success",
) -> None:
    st.session_state.notice = (
        message,
        kind,
    )


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title("Local RAG Assistant")

st.caption(
    "Ask questions using your local documents. "
    "All inference runs locally."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    documents = get_documents()

    st.subheader("Documents")

    if documents:
        st.caption(
            f"{len(documents)} document(s) loaded"
        )

        with st.expander(
            "Loaded Documents",
            expanded=False,
        ):
            for document in documents:
                st.write(f"• {document}")

    else:
        st.caption("No documents loaded")

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Add Documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        accept_multiple_files=True,
        key=(
            f"document_uploader_"
            f"{st.session_state.uploader_version}"
        ),
    )

    if st.button(
        "Add / Update Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        DOCS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = 0

        for uploaded_file in uploaded_files:
            safe_name = Path(
                uploaded_file.name
            ).name

            extension = Path(
                safe_name
            ).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                continue

            destination = (
                DOCS_DIR / safe_name
            )

            destination.write_bytes(
                uploaded_file.getbuffer()
            )

            saved += 1

        if saved:
            try:
                rag = st.session_state.get(
                    "rag"
                )

                if rag is None:
                    with st.spinner(
                        "Starting local models "
                        "and indexing documents..."
                    ):
                        ensure_rag()

                else:
                    with st.spinner(
                        "Indexing new or changed "
                        "documents..."
                    ):
                        stats = rag.refresh_index()

                    print(
                        f"[ui] Index refresh: {stats}"
                    )

                # Keep existing chat history.
                st.session_state[
                    "uploader_version"
                ] += 1

                set_notice(
                    f"{saved} document(s) added or updated. "
                    "Future answers will use the updated document set."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    "Could not update the "
                    f"document index: {error}"
                )

    # -----------------------------------------------------
    # Remove document
    # -----------------------------------------------------

    documents = get_documents()

    if documents:
        selected_document = st.selectbox(
            "Remove Document",
            documents,
        )

        if st.button(
            "Delete Selected Document",
            use_container_width=True,
        ):
            try:
                document_path = (
                    DOCS_DIR
                    / selected_document
                )

                if document_path.exists():
                    document_path.unlink()

                rag = st.session_state.get(
                    "rag"
                )

                if rag is not None:
                    with st.spinner(
                        "Updating document index..."
                    ):
                        rag.refresh_index()

                # Keep existing chat history.
                set_notice(
                    f"Deleted: {selected_document}. "
                    "Future answers will use the updated document set."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    "Could not delete the "
                    f"document: {error}"
                )

    # -----------------------------------------------------
    # Clear chat
    # -----------------------------------------------------

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "PDF · DOCX · TXT · MD"
    )


# ---------------------------------------------------------
# Notification
# ---------------------------------------------------------

if "notice" in st.session_state:
    message, kind = (
        st.session_state.pop(
            "notice"
        )
    )

    if kind == "success":
        st.success(message)
    else:
        st.info(message)


# ---------------------------------------------------------
# Require at least one document
# ---------------------------------------------------------

documents = get_documents()

if not documents:
    st.info(
        "Add at least one document from "
        "the sidebar to start the assistant."
    )
    st.stop()


# ---------------------------------------------------------
# Start RAG
# ---------------------------------------------------------

try:
    if "rag" not in st.session_state:
        with st.spinner(
            "Starting local models..."
        ):
            rag = ensure_rag()
    else:
        rag = st.session_state.rag

except Exception as error:
    st.error(
        "Could not start the "
        "Local RAG system."
    )

    st.code(
        str(error)
    )

    st.stop()


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):
            show_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching documents "
                "and generating answer..."
            ):
                answer, sources = (
                    rag.answer(
                        question
                    )
                )

            st.markdown(
                answer
            )

            show_sources(
                sources
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            error_message = (
                f"An error occurred: "
                f"{error}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )