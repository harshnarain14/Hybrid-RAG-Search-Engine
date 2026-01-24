from typing import Literal


QueryType = Literal["document", "web", "hybrid"]


def classify_query(query: str) -> QueryType:
    """
    Classify a user query into:
    - document: answerable from internal documents
    - web: requires real-time / latest info
    - hybrid: needs both documents + web
    """

    query_lower = query.lower()

    # Strong web signals
    web_keywords = [
        "latest",
        "recent",
        "current",
        "today",
        "news",
        "trend",
        "update",
        "this year",
        "this month",
        "now",
    ]

    # Hybrid signals
    hybrid_keywords = [
        "compare",
        "difference",
        "vs",
        "versus",
        "impact",
        "how does",
        "pros and cons",
    ]

    if any(keyword in query_lower for keyword in web_keywords):
        return "web"

    if any(keyword in query_lower for keyword in hybrid_keywords):
        return "hybrid"

    return "document"
