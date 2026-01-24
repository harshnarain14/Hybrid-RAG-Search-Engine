from typing import List
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.models.document_models import Document, DocumentChunk


def chunk_documents(documents: List[Document]) -> List[DocumentChunk]:
    """
    Split documents into overlapping chunks suitable for vector indexing.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[DocumentChunk] = []

    for doc in documents:
        split_texts = splitter.split_text(doc.content)

        for idx, text in enumerate(split_texts):
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                source_id=doc.source_id,
                source_type=doc.source_type,
                title=doc.title,
                content=text,
                chunk_index=idx,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                },
            )
            chunks.append(chunk)

    return chunks
