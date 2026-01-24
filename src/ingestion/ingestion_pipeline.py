from pathlib import Path
from typing import List, Optional

from src.config.settings import PDF_DIR, TEXT_DIR
from src.models.document_models import Document
from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.text_loader import load_text_files_from_directory
from src.ingestion.wikipedia_loader import load_wikipedia_pages


def ingest_documents(
    load_pdfs: bool = True,
    load_texts: bool = True,
    wikipedia_topics: Optional[List[str]] = None,
) -> List[Document]:
    """
    Ingest documents from multiple sources and return a unified list of Documents.
    """

    documents: List[Document] = []

    if load_pdfs:
        documents.extend(load_pdfs_from_directory(PDF_DIR))

    if load_texts:
        documents.extend(load_text_files_from_directory(TEXT_DIR))

    if wikipedia_topics:
        documents.extend(load_wikipedia_pages(wikipedia_topics))

    return documents
