from utils.helpers import get_config
from embeddings.provider import EmbeddingProvider
from models.schemas import SearchResponse
from retrieval.aggregator import group_by_document
from retrieval.ranker import rank_documents
from vector_store.store import VectorStore

_cfg = get_config()["retrieval"]


CHUNK_OVERFETCH = _cfg["chunk_overfetch"]
DEFAULT_TOP_K = _cfg["default_top_k"]


class Retriever:
    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ):
        self.provider = provider or EmbeddingProvider()
        self.store = store or VectorStore()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> SearchResponse:
        query_vector = self.provider.embed_query(query)

        hits = self.store.search(query_vector, top_k=top_k * CHUNK_OVERFETCH)

        grouped = group_by_document(hits)
        results = rank_documents(grouped, top_k)

        return SearchResponse(query=query, results=results)
