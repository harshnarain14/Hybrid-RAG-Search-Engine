from typing import List
import streamlit as st
from tavily import TavilyClient
from src.models.document_models import WebSearchResult


def get_tavily_client() -> TavilyClient:
    tavily_key = None

    # Streamlit Cloud
    if hasattr(st, "secrets") and "TAVILY_API_KEY" in st.secrets:
        tavily_key = st.secrets["TAVILY_API_KEY"]

    # Local fallback
    if not tavily_key:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(".env")
            tavily_key = env.get("TAVILY_API_KEY")
        except Exception:
            pass

    if not tavily_key:
        raise RuntimeError("❌ TAVILY_API_KEY not found in Streamlit Secrets or .env")

    return TavilyClient(api_key=tavily_key)


def tavily_search(query: str, max_results: int = 5) -> List[WebSearchResult]:
    client = get_tavily_client()
    response = client.search(query=query, max_results=max_results)

    if not response or "results" not in response:
        return []

    results: List[WebSearchResult] = []
    for item in response["results"]:
        results.append(
            WebSearchResult(
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                url=item.get("url", ""),
            )
        )

    return results
