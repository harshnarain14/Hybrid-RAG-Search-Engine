from typing import List

from langchain_community.document_loaders.wikipedia import WikipediaLoader

from src.models.document_models import Document
from src.preprocessing.cleaner import clean_text


def load_wikipedia_pages(pages: List[str]) -> List[Document]:
    """
    Load Wikipedia pages by topic name and convert them into Document objects.
    """

    documents: List[Document] = []

    for topic in pages:
        loader = WikipediaLoader(query=topic, load_max_docs=1)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            cleaned_text = clean_text(doc.page_content)

            document = Document(
                source_id=doc.metadata.get("source", topic),
                source_type="wikipedia",
                title=doc.metadata.get("title", topic),
                content=cleaned_text,
                metadata=doc.metadata,
            )

            documents.append(document)

    return documents
