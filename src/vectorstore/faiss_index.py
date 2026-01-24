import pickle
from typing import List

from langchain_community.vectorstores.faiss import FAISS

from src.config.settings import FAISS_INDEX_PATH, DEBUG
from src.models.document_models import DocumentChunk
from src.vectorstore.embeddings import get_embedding_model


def index_documents(chunks: List[DocumentChunk]) -> FAISS:
    """
    Create a FAISS index from document chunks and persist it to disk.
    """

    if DEBUG:
        print(f"🔹 Indexing {len(chunks)} chunks into FAISS")

    texts = [chunk.content for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    embedding_model = get_embedding_model()

    faiss_index = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
    )

    # Persist FAISS index
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(str(FAISS_INDEX_PATH) + ".pkl", "wb") as f:
        pickle.dump(faiss_index, f)

    if DEBUG:
        print("✅ FAISS index created and saved")

    return faiss_index


def load_faiss_index() -> FAISS | None:
    """
    Load FAISS index from disk if it exists.
    """

    index_file = str(FAISS_INDEX_PATH) + ".pkl"

    try:
        with open(index_file, "rb") as f:
            faiss_index = pickle.load(f)

        if DEBUG:
            print("✅ FAISS index loaded from disk")

        return faiss_index

    except FileNotFoundError:
        if DEBUG:
            print("⚠️ FAISS index not found")
        return None
