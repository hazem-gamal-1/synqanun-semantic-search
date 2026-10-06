import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from utils.helpers import get_config
from ingestion.chunker import DocumentChunker
from ingestion.loader import DocumentLoader
from models.schemas import SearchResponse
from retrieval.retriever import Retriever

_cfg = get_config()["retrieval"]

state: dict[str, Retriever] = {}
loader = DocumentLoader()
chunker = DocumentChunker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever()
    yield
    state.clear()


app = FastAPI(title="Legal Semantic Search", lifespan=lifespan)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(
        _cfg["default_top_k"],
        alias="topK",
        ge=1,
        le=_cfg["max_top_k"],
        description="Number of documents",
    ),
) -> SearchResponse:
    return state["retriever"].search(q, top_k=top_k)


@app.post("/documents")
def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a .pdf / .txt / .docx file and index it."""

    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in loader.LOADERS:
        raise HTTPException(400, f"Supported types: {', '.join(loader.LOADERS)}")

    retriever = state["retriever"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(file.file.read())
        document = loader.load(path)

    document.metadata["source"] = name 
    chunks = chunker.chunk(document)
    if not chunks:
        raise HTTPException(422, "No extractable text found in the file.")

    vectors = retriever.provider.embed_documents([c.text for c in chunks])
    retriever.store.upsert_chunks(chunks, vectors, document_title=document.title)

    return {"document_id": document.id, "title": document.title, "chunks": len(chunks)}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    state["retriever"].store.delete_document(document_id)
    return {"document_id": document_id, "deleted": True}