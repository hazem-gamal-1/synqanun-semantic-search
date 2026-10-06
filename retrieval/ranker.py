from utils.helpers import get_config
from models.schemas import SearchResult

_cfg = get_config()["ranking"]

SUPPORT_WEIGHT = _cfg["support_weight"]
MAX_CHUNKS_PER_DOCUMENT = _cfg["max_chunks_per_document"]


def document_score(chunk_scores: list[float]) -> float:
    best, rest = chunk_scores[0], chunk_scores[1:MAX_CHUNKS_PER_DOCUMENT]
    if not rest:
        return best
    return best + SUPPORT_WEIGHT * (sum(rest) / len(rest))


def rank_documents(grouped: dict[str, dict], top_k: int) -> list[SearchResult]:
    results = [
        SearchResult(
            document_id=document_id,
            title=entry["title"] or document_id,
            score=document_score([score for _, score in entry["hits"]]),
            matched_chunks=[
                chunk for chunk, _ in entry["hits"][:MAX_CHUNKS_PER_DOCUMENT]
            ],
        )
        for document_id, entry in grouped.items()
    ]

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
