from src.ingestion.ingestion_pipeline import ingest_documents

docs = ingest_documents(
    load_pdfs=True,
    load_texts=True,
    wikipedia_topics=["Retrieval-Augmented Generation"]
)

print(f"Total documents loaded: {len(docs)}")
for d in docs:
    print(d.source_type, "->", d.title)

