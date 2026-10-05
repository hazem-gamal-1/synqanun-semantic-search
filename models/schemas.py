from typing import Any

from pydantic import BaseModel, Field


class DocumentSegment(BaseModel):
    """A segment extracted from the original document."""

    text: str

    segment_index: int = Field(
        ...,
        ge=0,
        description="Position of the segment within the document",
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Page number when available",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata provided by the document loader",
    )


class Document(BaseModel):
    """Represents a complete source document."""

    id: str
    title: str

    segments: list[DocumentSegment]

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata",
    )


class DocumentChunk(BaseModel):
    """Represents a searchable chunk derived from a document."""

    id: str
    document_id: str
    text: str

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of the chunk within the document",
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Source page when available",
    )

    segment_index: int | None = Field(
        default=None,
        ge=0,
        description="Source segment from which the chunk originated",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-level metadata",
    )


class SearchResult(BaseModel):
    """Represents a document returned by semantic search."""

    document_id: str
    title: str
    score: float

    matched_chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="Chunks that contributed to the search result",
    )


class SearchResponse(BaseModel):
    """Represents the response returned by the search API."""

    query: str
    results: list[SearchResult]
