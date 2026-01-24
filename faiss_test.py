from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents
from src.vectorstore.faiss_index import index_documents, load_faiss_index

docs = ingest_documents(load_pdfs=True, load_texts=True)
chunks = chunk_documents(docs)

index_documents(chunks)

faiss_index = load_faiss_index()
print(faiss_index)
