
from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """Generates embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document chunks."""

        embeddings = self.model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query."""

        embedding = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()