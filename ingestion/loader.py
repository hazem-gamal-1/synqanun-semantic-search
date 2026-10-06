from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from functools import partial
from models.schemas import Document, DocumentSegment
import hashlib


class DocumentLoader:
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".txt": partial(TextLoader, encoding="utf-8"),
        ".docx": Docx2txtLoader,
    }

    def load(self, file_path: str | Path) -> Document:

        path = Path(file_path)
        extension = path.suffix.lower()
        loader_class = self.LOADERS.get(extension)
        loader = loader_class(str(path))
        loaded_documents = loader.load()
       

        segments = []
        for index, loaded_document in enumerate(loaded_documents):
            text = loaded_document.page_content.strip()

            if not text:
                continue

            metadata = dict(loaded_document.metadata)

            segments.append(
                DocumentSegment(
                    text=text,
                    segment_index=index,
                    page_number=self._extract_page_number(metadata),
                    metadata=metadata,
                )
            )

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        document_id = content_hash

        return Document(
            id=document_id,
            title=path.stem,
            segments=segments,
            metadata={
                "filename": path.name,
                "file_type": extension,
                "source": str(path),
            },
        )

    @staticmethod
    def _extract_page_number(
        metadata: dict,
    ) -> int | None:
        """
        Convert a zero-based page number from the loader
        into a human-readable one-based page number.
        """

        page = metadata.get("page")

        if page is None:
            return None

        try:
            return int(page) + 1
        except (TypeError, ValueError):
            return None
