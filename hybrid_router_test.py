from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents
from src.vectorstore.faiss_index import index_documents
from src.retrieval.hybrid_router import hybrid_retrieve

docs = ingest_documents(load_pdfs=True, load_texts=True)
chunks = chunk_documents(docs)
faiss_index = index_documents(chunks)

queries = [
    "Explain attention mechanism",
    "Latest news about large language models",
    "How does RAG compare with current LLM tools?",
]

for q in queries:
    result = hybrid_retrieve(q, faiss_index)
    print("\nQUERY:", q)
    print("TYPE:", result["query_type"])
    print("DOC CHUNKS:", len(result["document_chunks"]))
    print("WEB RESULTS:", len(result["web_results"]))
