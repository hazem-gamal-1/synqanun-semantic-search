# SynQanun — Semantic Search for Arabic Legal Documents

A semantic search pipeline over Arabic (Egyptian) legal documents. Documents are split into chunks, embedded, stored in a vector database, and searched by meaning rather than keywords. Search runs at the **chunk level**, but results are returned at the **document level**.

---

## Table of Contents

1. [Search Flow](#1-search-flow)
2. [Chunking Strategy](#2-chunking-strategy)
3. [Embedding Model & Vector Database](#3-embedding-model--vector-database)
4. [Ranking & Aggregation](#4-ranking--aggregation)
5. [How to Run](#5-how-to-run)
6. [API Documentation](#6-api-documentation)
7. [Example Output](#7-example-output)
8. [Architecture & Scalability](#8-architecture--scalability)

---

## 1. Search Flow

### Indexing

```
Upload file (.pdf / .txt / .docx)
        │
        ▼
  Loader      → extracts text per page, document_id = SHA-256 of file bytes
        │
        ▼
  Chunker     → splits each page into chunks (≤ 1200 chars, 150 overlap)
        │
        ▼
  Embedding   → 384-dim normalized vectors
        │
        ▼
  Qdrant      → one point per chunk (vector + document_id, page, text, metadata)
```

### Search

```
GET /search?q=...&topK=N
        │
        ▼
  Embed the query (same model)
        │
        ▼
  Qdrant: top (topK × 5) chunks by cosine similarity
        │
        ▼
  Group chunks by document_id → score each document
        │
        ▼
  Return topK documents with their matched chunks
```

---

## 2. Chunking Strategy

**Method:** page-aware recursive character splitting with overlap.

1. **Segment first.** Each PDF page is a segment. Chunks never cross page boundaries, so every chunk maps to a page number.
2. **Keep small segments whole.** A page of ≤ 1200 characters becomes a single chunk.
3. **Otherwise split recursively** with separators tried in order: `\n\n` → `\n` → `؛` → `؟` → `!` → `.` → `،` → space → character. This prefers paragraph and sentence boundaries, including Arabic punctuation.
4. **150-character overlap** between consecutive chunks, so a sentence cut at a boundary still appears intact in one of them.

**Traceability:** every chunk has the id `{document_id}_chunk_{chunk_index}` and stores `document_id`, `page_number`, `segment_index` and the source file metadata, so chunk-level hits always map back to their original document.

**Why this strategy**

- ~1200 characters is roughly one or two legal articles: enough context to keep an offence together with its penalty, small enough for a focused embedding.
- Splitting on Arabic punctuation keeps ideas together better than fixed-size cuts.
- Page-bounded chunks give reliable page citations.

---

## 3. Embedding Model & Vector Database

- **Embedding model:** `intfloat/multilingual-e5-small` (384 dimensions, multilingual with Arabic support)
- **Vector database:** Qdrant Cloud (cosine similarity)

---

## 4. Ranking & Aggregation

Chunk hits are grouped by `document_id`, then each document is scored:

```
document_score = best_chunk_score + 0.1 × mean(next best chunk scores)
```

- The best chunk dominates; extra matching chunks give a small boost.
- Up to 3 chunks per document are considered and returned.
- Documents are sorted by score and the top `topK` are returned.

---

## 5. How to Run

The vector database is hosted on Qdrant Cloud, so the only thing to run is the FastAPI server.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables in a .env file
#    QDRANT_URL=<your-qdrant-cloud-url>
#    QDRANT_API_KEY=<your-api-key>
#    CONFIG_PATH=config/config.yaml

# 3. Run the API
uvicorn api.main:app --reload --port 8000
```

Interactive docs (Swagger): http://localhost:8000/docs

---

## 6. API Documentation

Base URL: `http://localhost:8000`

### `GET /search`

Semantic search. Returns documents ranked by relevance.

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `q` | string | yes | — | Search query (Arabic or English) |
| `topK` | integer | no | `5` | Number of **documents** to return (1–50) |

**Response `200`**

```jsonc
{
  "query": "string",
  "results": [
    {
      "document_id": "string",
      "title": "string",
      "score": 0.957,
      "matched_chunks": [
        {
          "id": "string",
          "document_id": "string",
          "text": "string",
          "chunk_index": 0,
          "page_number": 1,
          "segment_index": 0,
          "metadata": {}
        }
      ]
    }
  ]
}
```

**Errors:** `422` if `q` is missing/empty or `topK` is outside 1–50.

### `POST /documents`

Upload and index a document (`multipart/form-data`, field `file`). Supported: `.pdf`, `.txt`, `.docx`.

```json
{ "document_id": "5d07aa2c7ba4...", "title": "Palestine-Penal-Code-Gaza-Egyptian-1939-Arabic", "chunks": 210 }
```

**Errors:** `400` unsupported file type, `422` no extractable text.

### `DELETE /documents/{document_id}`

Removes all chunks of a document from the index.

```json
{ "document_id": "5d07aa2c7ba4...", "deleted": true }
```

---

## 7. Example Output

**Request**

```
GET /search?q=حكم من اهمل في تنظيف المداخن&topK=2
```

```bash
curl -G "http://localhost:8000/search" \
  --data-urlencode "q=حكم من اهمل في تنظيف المداخن" \
  --data-urlencode "topK=2"
```

**Response** (chunk text and metadata shortened with `...`)

```json
{
  "query": "حكم من اهمل في تنظيف المداخن",
  "results": [
    {
      "document_id": "5d07aa2c7ba48715efbc6a55878443131e5d3b0b5765168fcbd60a7dab07152c",
      "title": "Palestine-Penal-Code-Gaza-Egyptian-1939-Arabic",
      "score": 0.9571207475,
      "matched_chunks": [
        {
          "id": "5d07aa2c...152c_chunk_175",
          "document_id": "5d07aa2c...152c",
          "text": "... مادة 360 ) 1 ( الحريق الناشئ من عدم تنظيف أو ترميم الفران أو المداخن أو المحلت الخرى التي توقد فيها النار ... أو بسبب إهمال أخر يعاقب عليه بالحبس مدة ل تزيد على شهر أ وبدفع غرامة ل تزيد على مائتي جنيه مصري. ...",
          "chunk_index": 175,
          "page_number": 67,
          "segment_index": 66,
          "metadata": {
            "filename": "Palestine-Penal-Code-Gaza-Egyptian-1939-Arabic.pdf",
            "file_type": ".pdf",
            "total_pages": 84,
            "page": 66,
            "page_label": "67"
          }
        },
        {
          "id": "5d07aa2c...152c_chunk_187",
          "document_id": "5d07aa2c...152c",
          "text": "2. من أهمل فى تنظيف أو إصلح المداخن أو الفران أو العامل التي تستعمل فيها النار 3. من كان موكل بالتحفظ على مجنون ... مادة 378 )1( يعاقب بغرامة ل تجاوز خمسين جنيها ...",
          "chunk_index": 187,
          "page_number": 71,
          "segment_index": 70,
          "metadata": {
            "filename": "Palestine-Penal-Code-Gaza-Egyptian-1939-Arabic.pdf",
            "page": 70,
            "page_label": "71"
          }
        },
        {
          "id": "5d07aa2c...152c_chunk_189",
          "document_id": "5d07aa2c...152c",
          "text": "5. من أطفأ نور الغاز ... 6. من تسبب بإهماله فى إتلف شيء من منقولت الغير ... مادة 379 ) 1 ( يعاقب بغرامة ل تجاوز خمسة وعشرين جنيها ...",
          "chunk_index": 189,
          "page_number": 72,
          "segment_index": 71,
          "metadata": {
            "filename": "Palestine-Penal-Code-Gaza-Egyptian-1939-Arabic.pdf",
            "page": 71,
            "page_label": "72"
          }
        }
      ]
    }
  ]
}
```

---

## 8. Architecture & Scalability

The project is built to be **clean, modular, and scalable**:

- **Modular, single-responsibility design.** Ingestion (loading, chunking), embeddings, vector store, and retrieval (search, aggregation, ranking) are separate packages, each doing one job. Any layer can be swapped (embedding model, vector DB, chunking method) without touching the others.
- **Clean separation of concerns.** The API layer only handles HTTP; business logic lives in the retriever, and all storage access goes through a single `VectorStore` class.
- **Configuration-driven.** Chunk size, overlap, top-K limits, ranking weights and model settings live in `config.yaml`, not in code.
- **Typed data models.** Pydantic schemas define every object (`Document`, `DocumentChunk`, `SearchResult`) and validate API responses.
- **Scalable storage and search.** Qdrant Cloud uses HNSW approximate nearest-neighbor search and supports sharding and replication as the corpus grows.
- **Efficient and idempotent indexing.** Batched upserts, content-hash document ids and deterministic point ids make re-uploads safe, with no duplicates.
- **Stateless API.** The service holds no index state, so multiple instances can run behind a load balancer against the same Qdrant cluster.
- **Fast filtering and deletion.** A payload index on `document_id` makes per-document operations efficient.