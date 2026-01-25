import streamlit as st
from pathlib import Path

from pypdf import PdfReader
from src.models.document_models import Document


from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents
from src.vectorstore.faiss_index import load_faiss_index, index_documents
from src.retrieval.hybrid_router import hybrid_retrieve
from src.rag.context_builder import build_rag_context
from src.rag.answer_generator import generate_answer

# ----------------------------
# Helpers
# ----------------------------
def docs_from_uploaded_files(uploaded_files):
    """
    Convert uploaded files (PDF / TXT) into LangChain Documents
    directly from memory (Streamlit Cloud safe).
    """
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
# Dark Theme (Readable & High Contrast)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #f9fafb;
    }
    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #f9fafb !important;
    }
    .stCaption {
        font-size: 16px !important;
        color: #cbd5f5 !important;
        margin-bottom: 2rem;
    }
    label, p, span {
        color: #e5e7eb !important;
        font-size: 15px;
    }
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
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white !important;
        border-radius: 14px;
        font-size: 15px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        border: none;
    }
    [data-testid="stTabs"] button {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #f9fafb !important;
        border-bottom: 2px solid #6366f1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Session State
# ----------------------------
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = load_faiss_index()

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

# ----------------------------
# Header
# ----------------------------
st.title("🧠 Hybrid RAG Search Engine")
st.caption("Chat with your documents and real-time web knowledge")

# ----------------------------
# File Upload (In-Memory)
# ----------------------------
uploaded_files = st.file_uploader(
    "📎 Attach documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state.files_uploaded = True
    st.success(f"{len(uploaded_files)} document(s) uploaded. They will be indexed.")

# ----------------------------
# Search Mode
# ----------------------------
search_mode = st.radio(
    "Search Mode",
    ["Document", "Web", "Hybrid"],
    horizontal=True,
)

# ----------------------------
# Chat Input
# ----------------------------
question = st.text_input(
    "Ask a question",
    placeholder="Ask something about your documents or the web...",
)

# ----------------------------
# Ask Button
# ----------------------------
if st.button("Ask") and question:
    with st.spinner("Thinking..."):

        # 🔹 Build FAISS index ONLY when needed
        if search_mode in ["Document", "Hybrid"]:
            if st.session_state.files_uploaded or st.session_state.faiss_index is None:

                if uploaded_files:
                    docs = docs_from_uploaded_files(uploaded_files)
                else:
                    docs = ingest_documents(load_pdfs=True, load_texts=True)

                chunks = chunk_documents(docs)
                st.session_state.faiss_index = index_documents(chunks)
                st.session_state.files_uploaded = False

                st.success(f"Indexed {len(docs)} documents ({len(chunks)} chunks)")

        # 🔹 Hybrid Retrieval
        result = hybrid_retrieve(
            query=question,
            faiss_index=st.session_state.faiss_index,
            search_mode=search_mode,
            top_k=5,
        )

        # 🔹 Build Context
        context, sources = build_rag_context(
            result["document_chunks"],
            result["web_results"],
        )

        # 🔹 Generate Answer
        answer = generate_answer(
            question=question,
            context=context,
            sources=sources,
        )

    # ----------------------------
    # Output Tabs
    # ----------------------------
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
