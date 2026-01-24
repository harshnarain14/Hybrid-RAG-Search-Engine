from src.web_search.tavily_client import tavily_search

results = tavily_search("Latest developments in RAG systems", max_results=3)

for r in results:
    print("TITLE:", r.title)
    print("URL:", r.url)
    print("SNIPPET:", r.snippet[:150])
    print("-" * 40)
