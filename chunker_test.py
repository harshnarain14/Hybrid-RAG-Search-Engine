from src.ingestion.ingestion_pipeline import ingest_documents
from src.preprocessing.chunker import chunk_documents

docs = ingest_documents(load_pdfs=True, load_texts=True)
chunks = chunk_documents(docs)

print(f"Documents: {len(docs)}")
print(f"Chunks: {len(chunks)}")

print(chunks[0].title)
print(chunks[0].content[:200])
print(chunks[0].metadata)