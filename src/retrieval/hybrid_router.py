from typing import Dict, Any, List
from src.models.document_models import DocumentChunk, WebSearchResult
from src.vectorstore.semantic_search import semantic_search
from src.web_search.tavily_client import tavily_search


def hybrid_retrieve(
    query: str,
    faiss_index,
    search_mode: str = "Hybrid",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Hybrid retrieval logic:
    - Document: FAISS only
    - Web: Tavily only
    - Hybrid: Both
    """

    document_chunks: List[DocumentChunk] = []
    web_results: List[WebSearchResult] = []

    # 📄 Document search
    if search_mode in ["Document", "Hybrid"] and faiss_index is not None:
        document_chunks = semantic_search(
            query=query,
            faiss_index=faiss_index,
            top_k=top_k,
        )

    # 🌐 Web search
    if search_mode in ["Web", "Hybrid"]:
        web_results = tavily_search(
            query=query,
            max_results=top_k,
        )

    return {
        "document_chunks": document_chunks,
        "web_results": web_results,
    }
