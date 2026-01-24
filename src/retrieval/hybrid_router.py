from typing import Dict, List, TypedDict

from langchain_community.vectorstores.faiss import FAISS

from src.models.document_models import DocumentChunk, WebSearchResult
from src.retrieval.query_classifier import classify_query
from src.vectorstore.semantic_search import semantic_search
from src.web_search.tavily_client import tavily_search


class RetrievalResult(TypedDict):
    query_type: str
    document_chunks: List[DocumentChunk]
    web_results: List[WebSearchResult]


def hybrid_retrieve(
    query: str,
    faiss_index: FAISS | None,
    top_k_docs: int = 5,
    top_k_web: int = 5,
) -> RetrievalResult:
    """
    Route query to document search, web search, or hybrid search.
    Always returns a structured RetrievalResult.
    """

    query_type = classify_query(query)

    document_chunks: List[DocumentChunk] = []
    web_results: List[WebSearchResult] = []

    # 📄 Document-only search
    if query_type == "document":
        if faiss_index is not None:
            document_chunks = semantic_search(
                faiss_index=faiss_index,
                query=query,
                top_k=top_k_docs,
            )

    # 🌐 Web-only search
    elif query_type == "web":
        web_results = tavily_search(
            query=query,
            max_results=top_k_web,
        )

    # 🔀 Hybrid search
    elif query_type == "hybrid":
        if faiss_index is not None:
            document_chunks = semantic_search(
                faiss_index=faiss_index,
                query=query,
                top_k=top_k_docs,
            )

        web_results = tavily_search(
            query=query,
            max_results=top_k_web,
        )

    return {
        "query_type": query_type,
        "document_chunks": document_chunks,
        "web_results": web_results,
    }
