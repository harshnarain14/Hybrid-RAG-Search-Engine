from typing import List

from langchain_community.vectorstores.faiss import FAISS

from src.config.settings import TOP_K_DOCUMENTS
from src.models.document_models import DocumentChunk


def semantic_search(
    faiss_index: FAISS,
    query: str,
    top_k: int = TOP_K_DOCUMENTS,
) -> List[DocumentChunk]:
    """
    Perform semantic search over FAISS index and return top-K DocumentChunks.
    """

    results = faiss_index.similarity_search(query, k=top_k)

    chunks: List[DocumentChunk] = []

    for res in results:
        chunk = DocumentChunk(
    chunk_id=str(res.metadata["chunk_id"]),
    source_id=str(res.metadata["source_id"]),
    source_type=str(res.metadata["source_type"]),
    title=str(res.metadata["title"]),
    content=res.page_content,
    chunk_index=int(res.metadata["chunk_index"]),
    metadata=res.metadata,
)

        chunks.append(chunk)

    return chunks
