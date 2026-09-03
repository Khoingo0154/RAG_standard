from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class QueryRequest:
    query: str
    file_id: Optional[str] = None
    top_k: int = 5


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_id: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    score: float
    filename: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    query_time_ms: float = 0.0
    intent: str = "DOCUMENT_QUERY"
