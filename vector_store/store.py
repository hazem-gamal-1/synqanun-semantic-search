import os
import uuid
from qdrant_client import QdrantClient, models
from models.schemas import DocumentChunk
from dotenv import load_dotenv
load_dotenv()

VECTOR_SIZE = 384
COLLECTION_NAME = "legal_chunks"


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

    def ensure_collection(self) -> None:
        """Create the collection (and payload indexes) if it does not exist."""

        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        document_title: str | None = None,
        batch_size: int = 128,
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
