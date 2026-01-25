import streamlit as st
from pathlib import Path

from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents
from src.vectorstore.faiss_index import load_faiss_index, index_documents
from src.retrieval.hybrid_router import hybrid_retrieve
from src.rag.context_builder import build_rag_context
from src.rag.answer_generator import generate_answer

def docs_from_uploaded_files(uploaded_files):
    docs = []

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif file.name.lower().endswith(".txt"):
            text = file.read().decode("utf-8")

        else:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_id": file.name,
                    "title": file.name,
                    "source_type": "upload",
                }
            )
        )

    return docs
# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Hybrid RAG Search Engine",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------
# POLISHED DARK THEME (HIGH CONTRAST)
# ----------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #f9fafb;
    }

    /* Main title */
    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #f9fafb !important;
        margin-bottom: 0.25rem;
    }

    /* Subtitle */
    .stCaption {
        font-size: 16px !important;
        color: #cbd5f5 !important;
        margin-bottom: 2rem;
    }

    /* Section labels */
    label, p, span {
        color: #e5e7eb !important;
        font-size: 15px;
    }

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background-color: #020617;
        border: 1px dashed #334155;
        border-radius: 14px;
        padding: 18px;
        color: #e5e7eb;
    }

    /* Remove white uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important;
    }

    /* Radio buttons */
    .stRadio label {
        font-size: 15px;
        font-weight: 500;
        color: #e5e7eb !important;
    }

    /* Text input */
    input[type="text"] {
        background-color: #020617 !important;
        color: #f9fafb !important;
        border-radius: 14px;
        border: 1px solid #334155;
        padding: 12px;
        font-size: 16px;
    }

    input::placeholder {
        color: #64748b !important;
    }

    /* Ask button */
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white !important;
        border-radius: 14px;
        font-size: 15px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        border: none;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #94a3b8 !important;
        font-size: 14px;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #f9fafb !important;
        border-bottom: 2px solid #6366f1;
    }

    /* Success & info boxes */
    .stAlert {
        background-color: #020617 !important;
        border-left: 4px solid #6366f1;
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Upload Directory
# ----------------------------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Session State
# ----------------------------
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = load_faiss_index()

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

# ----------------------------
# HEADER
# ----------------------------
st.title("🧠 Hybrid RAG Search Engine")
st.caption("Chat with your documents and real-time web knowledge")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "📎 Attach documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        with open(UPLOAD_DIR / file.name, "wb") as f:
            f.write(file.read())
    st.session_state.files_uploaded = True
    st.success("Documents uploaded. They will be indexed on your next question.")

# ----------------------------
# SEARCH MODE
# ----------------------------
search_mode: str = st.radio(
    "Search Mode",
    ["Document", "Web", "Hybrid"],
    horizontal=True,
) or "Hybrid"

# ----------------------------
# CHAT INPUT
# ----------------------------
question = st.text_input(
    "Ask a question",
    placeholder="Ask something about your documents or the web...",
)

# ----------------------------
# ASK
# ----------------------------
if st.button("Ask") and question:
    with st.spinner("Thinking..."):

        if search_mode in ["Document", "Hybrid"]:
            if st.session_state.files_uploaded or st.session_state.faiss_index is None:
                docs = ingest_documents(load_pdfs=True, load_texts=True)
                chunks = chunk_documents(docs)
                st.session_state.faiss_index = index_documents(chunks)
                st.session_state.files_uploaded = False

        result = hybrid_retrieve(
            query=question,
            faiss_index=st.session_state.faiss_index,
            search_mode=search_mode,
            top_k=5,
        )

        context, sources = build_rag_context(
            result["document_chunks"],
            result["web_results"],
        )

        answer = generate_answer(
            question=question,
            context=context,
            sources=sources,
        )

    tab1, tab2, tab3 = st.tabs(["💬 Answer", "📄 Documents", "🌐 Web"])

    with tab1:
        st.markdown(answer)

    with tab2:
        if result["document_chunks"]:
            for chunk in result["document_chunks"]:
                st.markdown(f"**{chunk.title} (Chunk {chunk.chunk_index})**")
                st.write(chunk.content)
                st.divider()
        else:
            st.info("No document evidence.")

    with tab3:
        if result["web_results"]:
            for web in result["web_results"]:
                st.markdown(f"**{web.title}**")
                st.write(web.snippet)
                st.markdown(web.url)
                st.divider()
        else:
            st.info("No web evidence.")
