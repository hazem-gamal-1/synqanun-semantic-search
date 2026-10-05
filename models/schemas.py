from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int = Field(..., ge=0)
    page_number: int | None = Field(default=None)
    source_reference: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    document_id: str
    title: str
    score: float
    matched_chunks: list[DocumentChunk] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
