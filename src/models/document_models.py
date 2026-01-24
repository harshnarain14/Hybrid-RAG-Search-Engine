from dataclasses import dataclass, field
from typing import Dict, Any


# --------------------------------------------------
# Base Document (PDF / Text / Wikipedia)
# --------------------------------------------------
@dataclass
class Document:
    source_id: str                 # unique identifier (file path / wiki url)
    source_type: str               # pdf | text | wikipedia | web
    title: str
    content: str                   # full cleaned text
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------
# Chunked Document (Stored in FAISS)
# --------------------------------------------------
@dataclass
class DocumentChunk:
    chunk_id: str                  # unique chunk id
    source_id: str
    source_type: str
    title: str
    content: str                   # chunk text
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------
# Real-Time Web Search Result (Temporary)
# --------------------------------------------------
@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str
    source_type: str = "web"


# --------------------------------------------------
# Answer Source (Used for Citations in UI)
# --------------------------------------------------
@dataclass
class AnswerSource:
    source_type: str               # doc | web
    title: str
    reference: str                 # chunk_id or URL
