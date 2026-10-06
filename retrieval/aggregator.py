from collections import defaultdict

from models.schemas import DocumentChunk


def group_by_document(hits: list[tuple[DocumentChunk, str | None, float]]) -> dict[str, dict]:

    
    grouped: dict[str, dict] = defaultdict(lambda: {"title": None, "hits": []})

    for chunk, title, score in hits:
        entry = grouped[chunk.document_id]
        entry["title"] = entry["title"] or title
        entry["hits"].append((chunk, score))

    for entry in grouped.values():
        entry["hits"].sort(key=lambda h: h[1], reverse=True)

    return grouped
