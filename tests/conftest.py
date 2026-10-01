"""Shared fixtures.

Two tiers of fixture, deliberately kept apart because they differ enormously in cost:

* :func:`spacy_doc` and :func:`make_context` build only spaCy views and hand-written
  coreference resolutions. They need ``en_core_web_sm`` but never FastCoref, so they run
  in seconds.
* :func:`extraction_result` additionally runs FastCoref over the whole corpus, which is
  the only genuinely slow thing in the suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from spacy.tokens import Span

from src.config import AppConfig
from src.coreference import (
    CoreferenceResolver,
    DeterministicCorefFallback,
    FastCorefResolver,
)
from src.document_loader import DocumentLoader
from src.entity_resolver import EntityRegistry
from src.logging_utils import configure_logging
from src.models import (
    Document,
    MentionResolution,
    MentionSource,
    ResolutionSource,
)
from src.nlp_processor import NLPProcessor
from src.rules import RelationExtractor
from src.rules.base import RuleContext
from src.triple_builder import TripleBuilder

configure_logging("WARNING")

ResolutionSpec = tuple[str, str, str, MentionSource]


@dataclass(frozen=True)
class ExtractionResult:
    """Everything S5/S6 produced, so tests can assert on any layer of the pipeline."""

    triples: tuple
    candidates: tuple
    skipped: tuple
    rejections: object


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return AppConfig.from_env()


@pytest.fixture(scope="session")
def registry() -> EntityRegistry:
    return EntityRegistry.load()


@pytest.fixture(scope="session")
def processor(config: AppConfig, registry: EntityRegistry) -> NLPProcessor:
    return NLPProcessor(config, registry.ruler_patterns())


@pytest.fixture(scope="session")
def documents(config: AppConfig) -> tuple[Document, ...]:
    return DocumentLoader(config).load()


def make_document(text: str, document_id: str = "unit") -> Document:
    return Document(
        document_id=document_id,
        filename=f"{document_id}.txt",
        text=text,
        character_count=len(text),
    )


def resolution_for_text(
    document: Document,
    mention_text: str,
    entity_id: str,
    label: str,
    mention_source: MentionSource = MentionSource.RULER,
    resolution_source: ResolutionSource = ResolutionSource.ALIAS,
) -> MentionResolution:
    """A resolution pinned to the *first* occurrence of ``mention_text``."""
    start = document.text.index(mention_text)
    return MentionResolution(
        mention=mention_text,
        resolved_to=entity_id,
        resolved_label=label,
        entity_id=entity_id,
        source=resolution_source,
        document_id=document.document_id,
        sentence_index=0,
        start_char=start,
        end_char=start + len(mention_text),
        mention_source=mention_source,
    )


def make_context(
    text: str,
    specs: tuple[ResolutionSpec, ...],
    processor: NLPProcessor,
    registry: EntityRegistry,
    sentence_index: int = 0,
) -> RuleContext:
    """Build a :class:`RuleContext` whose coreference resolutions are supplied by hand.

    Coreference is stubbed deliberately: these tests pin *span matching* and *parse shape*,
    so the resolution index is passed in explicitly instead of inferred. That keeps them
    independent of FastCoref, which is slow and whose behaviour is already covered by the
    coreference tests.
    """
    document = make_document(text)
    processed = processor.process(document)
    resolutions = {}
    for mention, entity_id, label, source in specs:
        resolution = resolution_for_text(document, mention, entity_id, label, source)
        resolutions[(resolution.start_char, resolution.end_char)] = resolution
    # `doc[int]` indexes tokens, not sentences, and `doc.sents` is a generator.
    sentence: Span = list(processed.doc.sents)[sentence_index]
    return RuleContext(
        processed=processed,
        sentence=sentence,
        sentence_index=sentence_index,
        resolutions=resolutions,
        doc_text=text,
        registry=registry,
    )


@pytest.fixture(scope="session")
def extraction_result(
    config: AppConfig,
    registry: EntityRegistry,
    processor: NLPProcessor,
    documents: tuple[Document, ...],
) -> ExtractionResult:
    """Run S3-S6 end to end over the real corpus. Slow: FastCoref inference on CPU."""
    resolver = CoreferenceResolver(
        config,
        registry,
        FastCorefResolver(config, processor.load()),
        DeterministicCorefFallback(registry),
    )

    pairs = []
    for document in documents:
        processed = processor.process(document)
        index = {(r.start_char, r.end_char): r for r in resolver.resolve(processed)}
        pairs.append((processed, index))

    extractor = RelationExtractor(
        enable_optional_relations=config.enable_optional_relations
    )
    candidates, skipped = extractor.extract(pairs, registry=registry)
    triples, rejections = TripleBuilder(registry).build(candidates)
    return ExtractionResult(triples, candidates, skipped, rejections)
