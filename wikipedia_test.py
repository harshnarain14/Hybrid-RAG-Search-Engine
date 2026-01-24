from src.ingestion.wikipedia_loader import load_wikipedia_pages

docs = load_wikipedia_pages(["Retrieval-Augmented Generation"])
print(docs[0].title)
print(docs[0].content[:300])

