from typing import Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(..., description="Unique identifier of the document")
    title: str = Field(..., description="Title of the document")
    text: str = Field(..., description="Full extracted text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique identifier of the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Text contained in the chunk")
    chunk_index: int = Field(..., ge=0, description="Position of the chunk in the document")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    document_id: str
    title: str
    score: float
    matched_chunks: list[DocumentChunk] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]