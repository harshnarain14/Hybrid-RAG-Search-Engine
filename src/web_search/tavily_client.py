from typing import List

from tavily import TavilyClient
from dotenv import dotenv_values

from src.models.document_models import WebSearchResult


def get_tavily_client() -> TavilyClient:
    env = dotenv_values(".env")
    tavily_key = env.get("TAVILY_API_KEY")

    if not tavily_key:
        raise RuntimeError("❌ TAVILY_API_KEY missing in .env file")

    return TavilyClient(api_key=tavily_key)


def tavily_search(
    query: str,
    max_results: int = 5,
) -> List[WebSearchResult]:
    """
    Perform a real-time web search using Tavily.
    Always returns a List[WebSearchResult].
    """

    client = get_tavily_client()

    response = client.search(
        query=query,
        max_results=max_results,
    )

    # 🔒 GUARANTEE: always return a list
    if not response or not isinstance(response, dict):
        return []

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        return []

    results: List[WebSearchResult] = []

    for item in raw_results:
        results.append(
            WebSearchResult(
                title=str(item.get("title", "")),
                snippet=str(item.get("content", "")),
                url=str(item.get("url", "")),
            )
        )

    return results
