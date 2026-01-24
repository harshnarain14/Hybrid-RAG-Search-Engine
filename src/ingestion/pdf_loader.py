from pathlib import Path
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader

from src.models.document_models import Document
from src.preprocessing.cleaner import clean_text


def load_pdfs_from_directory(pdf_dir: Path) -> List[Document]:
    """
    Load all PDF files from a directory and convert them into Document objects.
    """

    documents: List[Document] = []

    if not pdf_dir.exists():
        return documents

    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        full_text = "\n".join(page.page_content for page in pages)
        full_text = clean_text(full_text)

        document = Document(
            source_id=str(pdf_path),
            source_type="pdf",
            title=pdf_path.name,
            content=full_text,
            metadata={
                "file_name": pdf_path.name,
                "num_pages": len(pages),
            },
        )

        documents.append(document)

    return documents
