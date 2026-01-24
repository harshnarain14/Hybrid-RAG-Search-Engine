from src.retrieval.query_classifier import classify_query

queries = [
    "Explain attention mechanism",
    "Latest developments in GPT models",
    "How does RAG compare with current LLM tools?",
]

for q in queries:
    print(q, "->", classify_query(q))
