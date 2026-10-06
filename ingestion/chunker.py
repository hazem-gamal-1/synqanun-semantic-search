from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import get_config
from models.schemas import Document, DocumentChunk

_cfg = get_config()["chunking"]


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = _cfg["chunk_size"],
        chunk_overlap: int = _cfg["chunk_overlap"],
    ):

        self.chunk_size = chunk_size

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                "؛",
                "؟",
                "!",
                ".",
                "،",
                " ",
                "",
            ],
        )

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Chunk document segments while preserving provenance."""

        chunks = []

        for segment in document.segments:
            text = segment.text.strip()

            if not text:
                continue

            # Preserve the segment if it fits within the limit.
            if len(text) <= self.chunk_size:
                text_chunks = [text]
            else:
                text_chunks = self.splitter.split_text(text)

            for chunk_text in text_chunks:
                if not chunk_text.strip():
                    continue

                chunk_index = len(chunks)

                chunks.append(
                    DocumentChunk(
                        id=f"{document.id}_chunk_{chunk_index}",
                        document_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        page_number=segment.page_number,
                        segment_index=segment.segment_index,
                        metadata={
                            **document.metadata,
                            **segment.metadata,
                        },
                    )
                )

        return chunks
