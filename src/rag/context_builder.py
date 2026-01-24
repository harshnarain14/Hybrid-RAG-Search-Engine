from typing import List, Tuple

from src.models.document_models import DocumentChunk, WebSearchResult


def build_rag_context(
    document_chunks: List[DocumentChunk],
    web_results: List[WebSearchResult],
    max_chars: int = 6000,
) -> Tuple[str, List[str]]:
    """
    Build RAG context by combining document chunks and web results.
    """

    context_parts: List[str] = []
    sources: List[str] = []
    current_length = 0

    # -------- Document context --------
    for chunk in document_chunks:
        chunk_text = (
            f"[DOC]\n"
            f"Title: {chunk.title}\n"
            f"Content: {chunk.content}\n"
        )

        if current_length + len(chunk_text) > max_chars:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

        sources.append(
            f"[Doc] {chunk.title} – Chunk {chunk.chunk_index}"
        )

    # -------- Web context --------
    for result in web_results:
        web_text = (
            f"[WEB]\n"
            f"Title: {result.title}\n"
            f"Snippet: {result.snippet}\n"
            f"URL: {result.url}\n"
        )

        if current_length + len(web_text) > max_chars:
            break

        context_parts.append(web_text)
        current_length += len(web_text)

        sources.append(
            f"[Web] {result.title}"
        )

    context_text = "\n".join(context_parts)

    return context_text, sources
