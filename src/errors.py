"""Typed exception hierarchy for the Knowledge Graph MVP.

Rule E3 (docs/06_ENGINEERING_GUIDELINES.md): the pipeline raises specific types so the
caller can decide whether to abort (configuration) or degrade (bad data).
"""

from __future__ import annotations


class KnowledgeGraphError(Exception):
    """Root of the project's exception hierarchy."""


class ConfigurationError(KnowledgeGraphError):
    """The application is misconfigured. Fail fast -- rule E5."""


class ModelUnavailableError(KnowledgeGraphError):
    """A required NLP model or dependency could not be loaded."""


class CorpusError(KnowledgeGraphError):
    """The document corpus is missing, unreadable, or empty as a whole."""


class RelationExtractionError(KnowledgeGraphError):
    """A relation rule failed unexpectedly.

    Raised only for programming errors. Ordinary "this sentence has no relation"
    outcomes are not exceptions -- they are skipped-candidate diagnostics.
    """


class QueryError(KnowledgeGraphError):
    """A query could not be parsed or executed.

    Carries a user-facing message (rule E4) because the UI renders it verbatim.
    """

    def __init__(self, message: str, *, query: str = "", intent: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.query = query
        self.intent = intent


__all__ = [
    "KnowledgeGraphError",
    "ConfigurationError",
    "ModelUnavailableError",
    "CorpusError",
    "RelationExtractionError",
    "QueryError",
]
