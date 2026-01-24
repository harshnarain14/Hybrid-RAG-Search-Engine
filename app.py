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
    page_title="GA02 Hybrid RAG Search Engine",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------
# Dark Theme Styling
# ----------------------------
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: #eaeaea; }
        .stTextInput textarea { background-color: #1c1f26; color: white; }
        .stButton button { background-color: #3b82f6; color: white; }
        .stRadio > div { flex-direction: row; }
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
# App Header
# ----------------------------
st.title("🧠 Hybrid RAG Search Engine")
st.caption("Chat with your documents + live web intelligence")

# ----------------------------
# Chat Area (ChatGPT-like)
# ----------------------------
with st.container():

    # ➕ Attach files (Chat-style)
    uploaded_files = st.file_uploader(
        "➕ Attach documents (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for file in uploaded_files:
            file_path = UPLOAD_DIR / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())

        st.session_state.files_uploaded = True
        st.success("📄 Files attached. They will be indexed when you ask a question.")

    # Search mode toggle (inside chat flow)
    search_mode = st.radio(
        "Search Mode",
        options=["Document", "Web", "Hybrid"],
        horizontal=True,
    )

    # Chat input
    question = st.text_input(
        "Ask a question",
        placeholder="Ask something about your documents or the web...",
    )

    # Ask button
    if st.button("Ask") and question:

        with st.spinner("Thinking..."):

            # 🔁 Rebuild index if new files were uploaded
            if st.session_state.files_uploaded or st.session_state.faiss_index is None:
                docs = ingest_documents(load_pdfs=True, load_texts=True)
                chunks = chunk_documents(docs)
                st.session_state.faiss_index = index_documents(chunks)
                st.session_state.files_uploaded = False

            # Hybrid retrieval
            result = hybrid_retrieve(
                query=question,
                faiss_index=st.session_state.faiss_index,
            )

            # Build RAG context
            context, sources = build_rag_context(
                result["document_chunks"],
                result["web_results"],
            )

            # Generate answer
            answer = generate_answer(
                question=question,
                context=context,
                sources=sources,
            )

        # ----------------------------
        # Output Tabs
        # ----------------------------
        tab1, tab2, tab3 = st.tabs(
            ["💬 Answer", "📄 Document Evidence", "🌐 Web Evidence"]
        )

        with tab1:
            st.markdown(answer)

        with tab2:
            if result["document_chunks"]:
                for chunk in result["document_chunks"]:
                    st.markdown(
                        f"**{chunk.title} (Chunk {chunk.chunk_index})**"
                    )
                    st.write(chunk.content)
                    st.divider()
            else:
                st.info("No document evidence used.")

        with tab3:
            if result["web_results"]:
                for web in result["web_results"]:
                    st.markdown(f"**{web.title}**")
                    st.write(web.snippet)
                    st.markdown(web.url)
                    st.divider()
            else:
                st.info("No web evidence used.")
