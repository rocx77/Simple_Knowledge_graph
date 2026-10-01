"""Typed data model for the Knowledge Graph MVP.

Rule T1: the pipeline passes dataclasses, never untyped dicts. Dicts appear only at
the serialisation boundary and in transient UI display state.

Every dataclass documents its invariants, as required by rule T2. Records that are
produced and then only read are frozen (rule O4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """Closed set of node types. ``str`` base so values serialise to readable JSON."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    PROJECT = "PROJECT"
    COMPONENT = "COMPONENT"
    LITERAL = "LITERAL"

    @property
    def is_literal(self) -> bool:
        return self is EntityType.LITERAL


class RelationName(str, Enum):
    """The canonical relation vocabulary (specification section 10).

    Stored in exactly one canonical direction. Query-time symmetry is the query
    engine's concern, never the graph's.
    """

    FOUNDED = "founded"
    DEVELOPS = "develops"
    LEADS = "leads"
    FOCUSES_ON = "focuses_on"
    WORKS_AT = "works_at"
    WORKS_ON = "works_on"
    COLLABORATES_WITH = "collaborates_with"
    PARTNERS_WITH = "partners_with"
    DEPLOYED_AT = "deployed_at"
    USES = "uses"
    IMPROVES = "improves"
    INTEGRATED_INTO = "integrated_into"
    CUSTOMER_OF = "customer_of"
    CONTAINS = "contains"
    REUSES = "reuses"


class ResolutionSource(str, Enum):
    """How an anaphoric mention was resolved. Recorded for explainability."""

    FASTCOREF = "fastcoref"
    ALIAS = "alias"
    ANTECEDENT = "antecedent"
    NONE = "unresolved"


class MentionSource(str, Enum):
    """How a candidate mention was discovered (specification section 7)."""

    RULER = "entity_ruler"
    SPACY_NER = "spacy_ner"
    COMMON_NOUN = "common_noun_noun_chunk"
    PRONOUN = "pronoun"
    COREF = "coreference"


class QueryIntent(str, Enum):
    """Supported query intents. Unrestricted NLU is explicitly out of scope."""

    RELATION_LOOKUP = "relation_lookup"
    CONNECTION_PATH = "connection_path"
    UNSUPPORTED = "unsupported"


class QueryDirection(str, Enum):
    """Which edge direction a relation lookup should traverse."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BOTH = "both"


class ConfidenceBand(float, Enum):
    """Deterministic rule-confidence labels.

    These are heuristic labels, NOT statistical probabilities (charter invariant I7).
    The distinction is stated in the artifacts and in the UI.
    """

    DIRECT = 1.00
    COREF_OR_NORMALISED = 0.90
    INFERRED = 0.80

    @property
    def explanation(self) -> str:
        return {
            ConfidenceBand.DIRECT: "direct named-entity relation",
            ConfidenceBand.COREF_OR_NORMALISED: (
                "relation involving a resolved coreference or a deterministic "
                "semantic normalisation"
            ),
            ConfidenceBand.INFERRED: "deterministic semantic inference",
        }[self]


@dataclass(frozen=True, slots=True)
class Document:
    """One source document.

    Invariants:
        * ``document_id`` is the filename stem and is unique within a corpus.
        * ``text`` is never shared or concatenated with another document.
        * ``sha256`` pins the exact bytes, so corpus drift is detectable.
    """

    document_id: str
    filename: str
    text: str
    sha256: str = ""
    character_count: int = 0
    sentence_count: int = 0
    read_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityMention:
    """A single surface occurrence of something entity-like (specification section 6).

    Invariants:
        * ``0 <= start_char < end_char <= len(document_text)``.
        * ``sentence_index`` is the index of the sentence containing the span.
        * ``text == document_text[start_char:end_char]``.
    """

    text: str
    start_char: int
    end_char: int
    label: str
    sentence_index: int
    document_id: str
    token_index: int = -1
    root_index: int = -1
    source: MentionSource = MentionSource.SPACY_NER
    entity_id: str | None = None
    canonical_label: str | None = None
    entity_type: EntityType | None = None


@dataclass(frozen=True, slots=True)
class MentionResolution:
    """The outcome of coreference for one mention (specification sections 8 and 9).

    Invariants:
        * Both the original mention and the resolved entity are retained -- this is
          what makes coreference demonstrable rather than invisible.
        * ``source`` is ``NONE`` exactly when ``entity_id`` is ``None``.
        * An unresolved mention is never promoted to a node (charter invariant I3).
    """

    mention: str
    resolved_to: str | None
    resolved_label: str | None
    entity_id: str | None
    source: ResolutionSource
    document_id: str
    sentence_index: int
    start_char: int
    end_char: int
    coref_cluster: tuple[str, ...] = ()
    conflict: bool = False
    conflict_detail: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.entity_id is not None


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    """A real-world entity. Exactly one node per canonical entity.

    Invariants:
        * ``entity_id`` is ``"<type>:<slug>"`` (see ``ids.make_entity_id``).
        * ``canonical`` is True. ``LITERAL`` nodes are not canonical entities and use
          ``LiteralNode`` instead.
        * ``aliases`` always contains ``label``.
    """

    entity_id: str
    label: str
    type: EntityType
    aliases: tuple[str, ...] = ()
    description: str = ""
    canonical: bool = True

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if self.label not in self.aliases:
            object.__setattr__(self, "aliases", (self.label, *self.aliases))


@dataclass(frozen=True, slots=True)
class LiteralNode:
    """A non-entity object, permitted only for prepositional objects.

    The specification requires ``(Project Aurora, focuses_on, graph-based transaction
    analysis)``. That object is a noun phrase, not a named entity, so it gets its own
    node type rather than being forced into ORGANIZATION or similar.
    """

    entity_id: str
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """A proposed relation, before triple construction validates it.

    Invariants:
        * ``subject``/``object`` are *canonical labels* (or literal labels).
        * ``original_subject``/``original_object`` are the surface forms as written.
        * ``confidence`` comes from :class:`ConfidenceBand`, never a computed probability.
        * ``inferred`` is set by the rule, never derived from ``confidence``.
    """

    relation: RelationName
    subject: str
    object: str
    subject_original: str
    object_original: str
    document_id: str
    sentence_index: int
    sentence: str
    rule_id: str
    confidence: ConfidenceBand
    inferred: bool = False
    subject_id: str = ""
    object_id: str = ""
    subject_type: EntityType | None = None
    object_type: EntityType | None = None
    coref_resolved: bool = False
    context_entity: str = ""
    notes: str = ""

    @property
    def endpoints_resolved(self) -> bool:
        return bool(self.subject_id) and bool(self.object_id)


@dataclass(frozen=True, slots=True)
class SkippedCandidate:
    """A relation that was considered and deliberately not emitted (rule E2).

    Surfacing these is a feature, not a defect: it proves the extractor declines to
    fabricate facts. ``reason`` explains why.
    """

    rule_id: str
    relation: str
    reason: str
    document_id: str
    sentence_index: int
    sentence: str


@dataclass(frozen=True, slots=True)
class Triple:
    """A provenance-complete extracted fact (specification section 12).

    Invariants:
        * ``subject_id``/``object_id`` are populated; a triple always has two real endpoints.
        * ``document_id`` + ``sentence_index`` + ``sentence`` locate the evidence.
        * ``original_subject``/``original_object`` preserve what the text actually said.
        * ``confidence`` is a :class:`ConfidenceBand` label.
        * ``inferred`` relations are always lower confidence than stated ones.
    """

    triple_id: str
    subject: str
    subject_id: str
    relation: RelationName
    object: str
    object_id: str
    document_id: str
    sentence_index: int
    sentence: str
    original_subject: str
    original_object: str
    coref_resolved: bool
    inferred: bool
    confidence: ConfidenceBand
    rule_id: str
    context_entity: str = ""
    duplicate_count: int = 1

    @property
    def as_tuple(self) -> tuple[str, str, str]:
        """``(subject, relation, object)`` for assertions and display."""
        relation = getattr(self.relation, "value", self.relation)
        return (self.subject, str(relation), self.object)


@dataclass(frozen=True, slots=True)
class Query:
    """A parsed natural-language query (specification sections 17 and 20).

    Invariants:
        * ``entity`` holds a canonical label once resolved, never a raw alias.
        * ``intent == UNSUPPORTED`` is the safe terminal state; never raise for an
          unparseable query, return it with a user-facing message (rule E4).
    """

    raw: str
    intent: QueryIntent
    entity: str | None = None
    entity_id: str | None = None
    relation: RelationName | None = None
    direction: QueryDirection = QueryDirection.INCOMING
    target_entity: str | None = None
    target_id: str | None = None
    matched_alias: str = ""
    matched_template: str = ""
    confidence: float = 0.0
    message: str = ""

    @property
    def is_supported(self) -> bool:
        return self.intent is not QueryIntent.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class Evidence:
    """The supporting text for one answer (specification section 38).

    Invariants: always carries the document, the sentence, the original surface forms
    and the resolution decisions that produced the canonical entities.
    """

    document_id: str
    filename: str
    sentence: str
    sentence_index: int
    triple: Triple
    coref_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PathStep:
    """One hop in a connection path."""

    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    """The full outcome of executing a query (specification section 21).

    Invariants: ``answers`` is sorted and de-duplicated; ``matched_triples`` may be
    longer than ``answers`` when several triples justify the same answer.
    """

    query: Query
    answers: tuple[str, ...] = ()
    answer_ids: tuple[str, ...] = ()
    matched_triples: tuple[Triple, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    path: tuple[PathStep, ...] = ()
    message: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.answers or self.path)

    @property
    def answer_count(self) -> int:
        return len(self.answers)


@dataclass(frozen=True, slots=True)
class StageReport:
    """Per-stage outcome. Degradation is recorded here, never hidden (rule E2).

    Invariants: a stage that raised still produces a report, with ``ok=False`` and a
    populated ``error``.
    """

    name: str
    elapsed_seconds: float
    ok: bool
    counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything one pipeline run produced.

    Invariants:
        * ``triples`` is de-duplicated and every triple has both endpoints resolved.
        * ``skipped_candidates`` and ``unresolved_mentions`` are always present, even
          when empty, so a clean run is distinguishable from a crashed run.
    """

    documents: tuple[Document, ...]
    triples: tuple[Triple, ...]
    entities: tuple[CanonicalEntity, ...]
    literals: tuple[LiteralNode, ...]
    mentions: tuple[EntityMention, ...]
    resolutions: tuple[MentionResolution, ...]
    skipped_candidates: tuple[SkippedCandidate, ...]
    stage_reports: tuple[StageReport, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, str] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    built_at: str = ""

    # -- convenience accessors used by the graph builder and the UI ----------

    def triples_for(self, entity_id: str) -> tuple[Triple, ...]:
        return tuple(t for t in self.triples if t.subject_id == entity_id or t.object_id == entity_id)

    def entity_by_id(self, entity_id: str) -> CanonicalEntity | None:
        return next((e for e in self.entities if e.entity_id == entity_id), None)

    def entity_by_label(self, label: str) -> CanonicalEntity | None:
        lowered = label.strip().lower()
        return next((e for e in self.entities if e.label.lower() == lowered), None)

    def mention_count(self, entity_id: str) -> int:
        return sum(1 for m in self.mentions if m.entity_id == entity_id)

    def documents_for(self, entity_id: str) -> tuple[str, ...]:
        return tuple(sorted({m.document_id for m in self.mentions if m.entity_id == entity_id}))

    @property
    def entity_count(self) -> int:
        return len(self.entities) + len(self.literals)

    @property
    def relation_count(self) -> int:
        return len(self.triples)


__all__ = [
    "EntityType",
    "RelationName",
    "ResolutionSource",
    "MentionSource",
    "QueryIntent",
    "QueryDirection",
    "ConfidenceBand",
    "Document",
    "EntityMention",
    "MentionResolution",
    "CanonicalEntity",
    "LiteralNode",
    "RelationCandidate",
    "SkippedCandidate",
    "Triple",
    "Query",
    "Evidence",
    "PathStep",
    "QueryResult",
    "StageReport",
    "PipelineResult",
]
