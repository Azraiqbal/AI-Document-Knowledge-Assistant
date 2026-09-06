import streamlit as st

from src.config import settings
from src.services.document_service import DocumentService
from src.services.rag_service import RAGService


st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="📚",
    layout="wide",
)


st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .status-box {
            padding: 0.9rem;
            border-radius: 0.7rem;
            background-color: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            margin-bottom: 1rem;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 0.6rem;
            background-color: #f8fafc;
            border-left: 4px solid #6366f1;
            margin-bottom: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session():
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = None

    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

    if "document_info" not in st.session_state:
        st.session_state.document_info = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


initialize_session()

document_service = DocumentService()


with st.sidebar:
    st.title("📄 Document Panel")

    st.write(
        "Upload a document and process it before asking questions."
    )

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX and TXT",
    )

    process_button = st.button(
        "⚙️ Process Document",
        type="primary",
        use_container_width=True,
    )

    if process_button:
        if uploaded_file is None:
            st.warning("Please upload a document first.")

        else:
            try:
                with st.spinner(
                    "Extracting text and creating embeddings..."
                ):
                    file_bytes = uploaded_file.getvalue()

                    result = document_service.process_document(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                    )

                    rag_service = RAGService()
                    rag_service.index_document(result["chunks"])

                    st.session_state.rag_service = rag_service
                    st.session_state.document_processed = True
                    st.session_state.document_info = {
                        "filename": result["filename"],
                        "pages": result["page_count"],
                        "chunks": result["chunk_count"],
                    }
                    st.session_state.chat_history = []

                st.success("Document processed successfully!")

            except ValueError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "The document could not be processed. "
                    "Please check the file and try again."
                )

    if st.session_state.document_processed:
        document_info = st.session_state.document_info

        st.markdown(
            f"""
            <div class="status-box">
                <strong>Ready for questions</strong><br>
                File: {document_info["filename"]}<br>
                Pages: {document_info["pages"]}<br>
                Chunks: {document_info["chunks"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button(
        "🗑️ Clear Document & Chat",
        use_container_width=True,
    ):
        if st.session_state.rag_service is not None:
            st.session_state.rag_service.clear_document()

        st.session_state.rag_service = None
        st.session_state.document_processed = False
        st.session_state.document_info = None
        st.session_state.chat_history = []

        st.rerun()


st.markdown(
    '<p class="main-title">📚 AI Document Knowledge Assistant</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="subtitle">
        Upload your document and receive accurate answers with
        source references using Retrieval-Augmented Generation.
    </p>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.document_processed:
    st.info(
        "Upload and process a PDF, DOCX or TXT document "
        "from the sidebar to begin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 Upload")
        st.write("Upload PDF, DOCX or TXT study material.")

    with col2:
        st.subheader("🔍 Retrieve")
        st.write("The system finds the most relevant document sections.")

    with col3:
        st.subheader("💬 Ask")
        st.write("Get document-grounded answers with references.")

else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("View sources and references"):
                    for number, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        relevance = max(
                            0,
                            min(100, source["score"] * 100),
                        )

                        st.markdown(
                            f"""
                            **Source {number}:**
                            `{source["source"]}` —
                            Page {source["page"]} —
                            Relevance {relevance:.1f}%
                            """
                        )

                        st.code(
                            source["text"],
                            language=None,
                            wrap_lines=True,
                        )

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:
        st.session_state.chat_history.append(
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
                    "Searching the document and generating an answer..."
                ):
                    response = (
                        st.session_state.rag_service.ask_question(
                            question
                        )
                    )

                st.markdown(response["answer"])

                if response["sources"]:
                    with st.expander(
                        "View sources and references"
                    ):
                        for number, source in enumerate(
                            response["sources"],
                            start=1,
                        ):
                            relevance = max(
                                0,
                                min(100, source["score"] * 100),
                            )

                            st.markdown(
                                f"""
                                **Source {number}:**
                                `{source["source"]}` —
                                Page {source["page"]} —
                                Relevance {relevance:.1f}%
                                """
                            )

                            st.code(source["text"], language=None, wrap_lines=True)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )

            except ValueError as error:
                st.error(str(error))

            except RuntimeError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "Something went wrong while generating the answer. "
                    "Please try again."
                )