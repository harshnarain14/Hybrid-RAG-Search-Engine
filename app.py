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
# Dark Theme (forced)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    [data-testid="stTextInput"] input {
        background-color: #1c1f26;
        color: white;
        border-radius: 12px;
        border: 1px solid #2a2f3a;
    }
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border-radius: 12px;
        font-weight: 600;
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
# UI
# ----------------------------
st.title("🧠 Hybrid RAG Search Engine")
st.caption("Chat with documents + live web search")

uploaded_files = st.file_uploader(
    "➕ Attach documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    for file in uploaded_files:
        with open(UPLOAD_DIR / file.name, "wb") as f:
            f.write(file.read())
    st.session_state.files_uploaded = True
    st.success("Files uploaded successfully.")

search_mode = st.radio(
    "Search Mode",
    ["Document", "Web", "Hybrid"],
    horizontal=True,
)

question = st.text_input("Ask a question")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):

        # 🚨 Build FAISS ONLY if needed
        if search_mode in ["Document", "Hybrid"]:
            if st.session_state.files_uploaded or st.session_state.faiss_index is None:
                docs = ingest_documents(load_pdfs=True, load_texts=True)
                chunks = chunk_documents(docs)
                st.session_state.faiss_index = index_documents(chunks)
                st.session_state.files_uploaded = False

        result = hybrid_retrieve(
            query=question,
            faiss_index=st.session_state.faiss_index,
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
            for c in result["document_chunks"]:
                st.markdown(f"**{c.title} (Chunk {c.chunk_index})**")
                st.write(c.content)
        else:
            st.info("No document evidence.")

    with tab3:
        if result["web_results"]:
            for w in result["web_results"]:
                st.markdown(f"**{w.title}**")
                st.write(w.snippet)
                st.markdown(w.url)
        else:
            st.info("No web evidence.")
