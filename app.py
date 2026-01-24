import streamlit as st
from pathlib import Path

from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents
from src.vectorstore.faiss_index import load_faiss_index, index_documents
from src.retrieval.hybrid_router import hybrid_retrieve
from src.rag.context_builder import build_rag_context
from src.rag.answer_generator import generate_answer

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Hybrid RAG Search Engine",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------
# FORCE HIGH-CONTRAST DARK THEME
# ----------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #0b0f14;
        color: #ffffff;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Normal text */
    p, span, label, div {
        color: #e5e7eb !important;
        font-size: 15px;
    }

    /* Caption */
    .stCaption {
        color: #9ca3af !important;
    }

    /* Text input */
    input[type="text"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px;
        border: 1px solid #374151;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111827;
        border: 1px dashed #374151;
        border-radius: 12px;
        padding: 16px;
        color: #e5e7eb;
    }

    /* Radio buttons */
    .stRadio label {
        color: #e5e7eb !important;
        font-weight: 500;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white !important;
        border-radius: 12px;
        font-weight: 600;
        border: none;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #9ca3af !important;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #3b82f6;
    }

    /* Info boxes */
    .stAlert {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border-left: 4px solid #3b82f6;
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
# UI HEADER
# ----------------------------
st.title("🧠 Hybrid RAG Search Engine")
st.caption("Chat with documents + live web search")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "➕ Attach documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        with open(UPLOAD_DIR / file.name, "wb") as f:
            f.write(file.read())
    st.session_state.files_uploaded = True
    st.success("Files uploaded successfully.")

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
# ASK BUTTON
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

    tab1, tab2, tab3 = st.tabs(["💬 Answer", "📄 Docs", "🌐 Web"])

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
