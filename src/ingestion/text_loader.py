from pathlib import Path
from typing import List

from langchain_community.document_loaders.text import TextLoader

from src.models.document_models import Document
from src.preprocessing.cleaner import clean_text


def load_text_files_from_directory(text_dir: Path) -> List[Document]:
    """
    Load all .txt and .md files from a directory and convert them into Document objects.
    """

    documents: List[Document] = []

    if not text_dir.exists():
        return documents

    for file_path in text_dir.glob("*"):
        if file_path.suffix.lower() not in [".txt", ".md"]:
            continue

        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            cleaned_text = clean_text(doc.page_content)

            document = Document(
                source_id=str(file_path),
                source_type="text",
                title=file_path.name,
                content=cleaned_text,
                metadata={
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                },
            )

            documents.append(document)

    return documents
