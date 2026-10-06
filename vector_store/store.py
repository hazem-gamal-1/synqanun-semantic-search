import os
import uuid

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

from config import get_config
from models.schemas import DocumentChunk

load_dotenv()

_emb = get_config()["embedding"]
_cfg = get_config()["vector_store"]

VECTOR_SIZE = _emb["vector_size"]
COLLECTION_NAME = _cfg["collection_name"]
DISTANCE = models.Distance[_cfg["distance"].upper()]
BATCH_SIZE = _cfg["upsert_batch_size"]

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)


class VectorStore:
    def __init__(
        self,
        qdrant_client: QdrantClient = client,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.ensure_collection()

    # ------------------------------------------------------------------ setup

    def ensure_collection(self) -> None:
        """Create the collection (and payload indexes) if it does not exist."""

        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=DISTANCE,
            ),
        )

        # Needed for fast delete/filter by document.
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # ----------------------------------------------------------------- writes

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant IDs must be UUIDs or ints; derive a stable UUID from the chunk id."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        document_title: str | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> int:

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must match"
            )

        points = [
            models.PointStruct(
                id=self._point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_title": document_title
                    or chunk.metadata.get("title")
                    or chunk.metadata.get("filename"),
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "segment_index": chunk.segment_index,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

        return len(points)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to a document (e.g. before re-indexing)."""

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    # ------------------------------------------------------------------ reads

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        score_threshold: float | None = None,
    ) -> list[tuple[DocumentChunk, str | None, float]]:

 
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            chunk = DocumentChunk(
                id=payload["chunk_id"],
                document_id=payload["document_id"],
                text=payload["text"],
                chunk_index=payload["chunk_index"],
                page_number=payload.get("page_number"),
                segment_index=payload.get("segment_index"),
                metadata=payload.get("metadata", {}),
            )
            results.append((chunk, payload.get("document_title"), point.score))

        return results