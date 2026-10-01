"""Stage 1 -- document ingestion.

Loads every ``.txt`` file under the configured corpus directory into
:class:`~src.models.Document` records. Documents are never concatenated: FastCoref is
run per document, so cross-document coreference is out of scope and merging text would
silently change the results (specification section 5).

Degradation contract (rule E2): an individual unreadable or empty file produces a
warning and is skipped; the run continues. An empty *corpus* is a configuration error
and does stop the run (rule E5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import AppConfig
from .errors import CorpusError
from .logging_utils import get_logger
from .models import Document

logger = get_logger(__name__)


class DocumentLoader:
    """Loads and validates the demonstration corpus."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load(self) -> tuple[Document, ...]:
        """Load every document in the corpus directory, in stable filename order."""
        docs_dir = self._config.docs_dir

        if not docs_dir.is_dir():
            raise CorpusError(
                f"Corpus directory not found: {docs_dir}\n"
                f"  Run 'python scripts/verify_corpus.py' to inspect the corpus, or set "
                f"KG_DOCS_DIR to point at a directory containing .txt documents."
            )

        paths = sorted(docs_dir.glob("*.txt"))
        if not paths:
            raise CorpusError(
                f"Corpus directory contains no .txt files: {docs_dir}\n"
                f"  Expected the five specification documents (doc_01_company.txt ... doc_05_customer.txt)."
            )

        documents: list[Document] = []
        for path in paths:
            document = self._load_one(path)
            if document is not None:
                documents.append(document)

        if not documents:
            raise CorpusError(
                f"Every document in {docs_dir} was unreadable or empty; nothing to process."
            )

        logger.info("Loaded %d document(s) from %s", len(documents), docs_dir)
        return tuple(documents)

    def _load_one(self, path: Path) -> Document | None:
        warnings: list[str] = []
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            text = raw.decode("utf-8", errors="replace")
            warnings.append(f"invalid utf-8 byte at offset {exc.start}; decoded with replacement")

        if "�" in text:
            warnings.append("text contains a Unicode replacement character")

        if not text.strip():
            logger.warning("Skipping %s: file is empty or whitespace only", path.name)
            return None

        logger.debug("Read %s (%d bytes, sha256=%s)", path.name, len(raw), digest[:12])
        return Document(
            document_id=path.stem,
            filename=path.name,
            text=text,
            sha256=digest,
            character_count=len(text),
            read_warnings=tuple(warnings),
        )


__all__ = ["DocumentLoader"]
