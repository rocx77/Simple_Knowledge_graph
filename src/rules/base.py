"""Shared machinery for relation rules.

Every rule needs the same four things: find a predicate, find its two arguments,
resolve both to canonical entities, and record what was skipped. Centralising that
here is what keeps each rule file down to the linguistic pattern it actually encodes
(guideline M4 -- all syntactic knowledge in one place).

The helpers encode the parser facts measured on the real corpus, each with the
observation that forced it. They are cited by number from the dependency probe:

* **P1** the passive agent preposition has dep ``agent``, not ``prep``.
* **P3** ``leading`` in an appositive is dep ``amod``, so verbs are found by lemma,
  never by dependency label alone.
* **P4** a preposition may attach to the *object* rather than the verb
  (``with -> contract``, ``for -> work``), so prepositions are searched over the whole
  subtree.
* **P6** ``plans to integrate`` puts the real predicate in an ``xcomp``.
* **P7** a reduced relative carries no subject, so the subject comes from the appositive
  host.
* **P9** an object must be expanded to its noun chunk, or ``analysis`` is emitted
  instead of ``graph-based transaction analysis``.
* **P8** a reporting verb blocks everything beneath it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models import (
    ConfidenceBand,
    EntityMention,
    EntityType,
    MentionResolution,
    RelationCandidate,
    RelationName,
    SkippedCandidate,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.tokens import Span, Token

    from ..entity_resolver import EntityRegistry
    from ..nlp_processor import ProcessedDocument

#: Verbs whose complement clause reports speech or belief rather than asserting fact.
#: Without this guard "Arun Mehta said the extension will reuse ..." yields
#: (Arun Mehta, reuses, Graph Matching Module), attributing an opinion as an action
#: (parser fact P8).
REPORTING_VERBS = frozenset(
    {
        "say", "report", "state", "claim", "note", "mention", "announce", "argue",
        "believe", "think", "assert", "explain", "write", "tell", "suggest", "declare",
    }
)

#: Dependency labels through which a verb's subject is found.
_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "expl"})

#: Dependency labels that introduce a prepositional object.
_PREP_DEPS = frozenset({"prep", "agent"})

#: Generic nominal heads that must never become a relation endpoint.
_NOT_A_RELATION_ENDPOINT = frozenset(
    {
        "work", "alert", "transaction", "branch", "pilot", "contract", "infrastructure",
        "analysis", "resolution", "detection", "time", "fraud", "study", "research",
        "year", "team", "engine",
    }
)


@dataclass(frozen=True, slots=True)
class Argument:
    """One resolved relation endpoint.

    Invariants:
        * ``label`` is the canonical entity (or literal) label.
        * ``surface`` is what the document actually said, preserved for provenance.
        * ``mention`` is the anaphor that was resolved, if the surface was anaphoric.
    """

    entity_id: str
    label: str
    surface: str
    entity_type: EntityType | None
    was_coref_resolved: bool = False
    mention: EntityMention | None = None
    resolution: MentionResolution | None = None

    @property
    def is_literal(self) -> bool:
        return self.entity_type is EntityType.LITERAL


@dataclass(slots=True)
class RuleContext:
    """Everything a rule may inspect for one sentence.

    Holds the live spaCy ``Span`` (rules need real dependency trees) plus an index from
    character span to coreference resolution, so a rule can ask "is this noun phrase an
    entity the pipeline already resolved?" instead of re-deriving it.
    """

    processed: ProcessedDocument
    sentence: Span
    sentence_index: int
    resolutions: Mapping[tuple[int, int], MentionResolution] = field(default_factory=dict)
    doc_text: str = ""
    #: The canonical registry, consulted for entity types and for the LITERAL endpoint
    #: that ``focuses_on`` legitimately takes.
    registry: EntityRegistry | None = None

    # -- token access -------------------------------------------------------

    @property
    def text(self) -> str:
        return self.sentence.text

    @property
    def tokens(self) -> list[Token]:
        return list(self.sentence)

    def subject_tokens(self, verb: Token) -> list[Token]:
        """Subject candidates for ``verb``.

        Falls back to the appositive host when the verb is a reduced relative, which has
        no subject of its own (parser fact P7).

        Note: spaCy builds a fresh Python wrapper on every ``token.head`` access, so
        identity comparison (``t.head is verb``) is always false and would silently
        discard every subject. Token *indices* are compared instead.
        """
        own = [t for t in verb.subtree if t.dep_ in _SUBJECT_DEPS and t.head.i == verb.i]
        if own:
            return own
        # Reduced relative: "the data scientist leading Project Aurora" parses `leading`
        # as an `amod` modifier of Project Aurora, which is itself an `appos` on Nila Rao.
        # The subject is therefore the appositive host's *head*. Rao's own dep in the
        # larger sentence is `pobj` (of "with"), which is irrelevant -- inside the
        # appositive it is a full noun phrase, and that is the role being asked for.
        host = verb.head
        if host.dep_ == "appos" and host.head.i != host.i:
            return [host.head]
        return []

    def direct_object(self, verb: Token) -> Token | None:
        for child in verb.children:
            if child.dep_ in {"dobj", "attr", "oprd"}:
                return child
        return None

    def preposition(self, verb: Token, preps: Iterable[str]) -> tuple[Token, Token] | None:
        """Find ``(preposition, object)`` for one of ``preps`` anywhere in the subtree.

        Walking the whole subtree rather than only direct children is required by
        parser fact P4: in "renewed its contract **with** Aether Analytics" the
        preposition attaches to ``contract``, not to ``renewed``.
        """
        wanted = {p.casefold() for p in preps}
        for token in verb.subtree:
            if token.dep_ in _PREP_DEPS and token.lower_ in wanted:
                pobj = self._object_of_prep(token)
                if pobj is not None:
                    return (token, pobj)
        return None

    def prepositions(self, verb: Token, preps: Iterable[str]) -> list[tuple[Token, Token]]:
        """All ``(preposition, object)`` pairs for ``preps``, in subtree order."""
        wanted = {p.casefold() for p in preps}
        found: list[tuple[Token, Token]] = []
        for token in verb.subtree:
            if token.dep_ in _PREP_DEPS and token.lower_ in wanted:
                pobj = self._object_of_prep(token)
                if pobj is not None:
                    found.append((token, pobj))
        return found

    @staticmethod
    def _object_of_prep(prep: Token) -> Token | None:
        for child in prep.children:
            if child.dep_ in {"pobj", "obj"}:
                return child
            if child.dep_ in {"nsubj", "nsubjpass"}:
                # "with whom they partnered" -- rare, but do not crash.
                return child
        return None

    def noun_chunk_of(self, token: Token) -> Span | None:
        """The noun chunk headed by ``token``, or containing it (parser fact P9).

        Compared by token index, never by object identity: spaCy hands out a new wrapper
        object per access, so ``chunk.root is token`` is always false.
        """
        for chunk in self.sentence.noun_chunks:
            if chunk.root.i == token.i or chunk.start <= token.i < chunk.end:
                return chunk
        return None

    def nominal_text(self, token: Token) -> str:
        """Human-readable nominal span for evidence: ``analysis`` -> ``graph-based ... analysis``."""
        chunk = self.noun_chunk_of(token)
        return chunk.text if chunk is not None else token.text

    def canonical_entity(self, entity_id: str) -> Any | None:
        """The registry record for an id, canonical entity or literal."""
        return self.registry.record(entity_id) if self.registry else None

    def argument_by_id(self, entity_id: str, *, allow_literal: bool = False) -> Argument | None:
        """Build an :class:`Argument` from a canonical id rather than a token span.

        Needed where the registry knows an entity the sentence never names directly, and
        to supply the LITERAL endpoint that ``focuses_on`` legitimately takes.
        """
        record = self.canonical_entity(entity_id)
        if record is None:
            return None
        entity_type = self.registry.type_of(entity_id) if self.registry else None
        label = getattr(record, "label", entity_id)
        if entity_type is EntityType.LITERAL and not allow_literal:
            return None
        return Argument(
            entity_id=entity_id,
            label=label,
            surface=label,
            entity_type=entity_type,
            was_coref_resolved=False,
        )

    def argument_for_own_token(self, token: Token) -> Argument | None:
        """Resolve a token using its **own** span before falling back to its noun chunk.

        Needed for possessives. In ``Aurora's graph matching module`` the whole thing is
        one noun chunk, but the possessor is ``Aurora`` alone; using the chunk would
        record ``original_subject="Aurora's graph matching module"`` for a ``contains``
        triple and put the containment inside its own subject.
        """
        if (arg := self.argument_for_span(token.idx, token.idx + len(token.text))) is not None:
            return arg
        return self.argument_for_token(token)

    # -- coreference lookup -------------------------------------------------

    def resolution_for(self, start_char: int, end_char: int) -> MentionResolution | None:
        """Find the best resolution for a char span.

        Matching is by **overlap**, not containment, because the two sides disagree about
        span boundaries in both directions:

        * the noun chunk includes the determiner -- ``the Entity Resolution Engine`` --
          while the ruler mention the coreference stage actually resolved is
          ``Entity Resolution Engine``, so the mention is contained in the requested span;
        * conversely, ``Aurora's graph matching module`` is one chunk while the possessor
          ``Aurora`` is a proper sub-span of it.

        Containment alone would miss both. Among overlapping resolutions the one sharing
        the most characters wins, which selects the specific mention rather than a
        neighbouring one.
        """
        if (exact := self.resolutions.get((start_char, end_char))) is not None:
            return exact

        best: tuple[int, tuple[int, int]] | None = None
        for span in self.resolutions:
            overlap = min(span[1], end_char) - max(span[0], start_char)
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, span)
        if best is None:
            return None
        return self.resolutions[best[1]]

    def argument_for_token(
        self, token: Token, *, allow_literal: bool = False, literal: bool = False
    ) -> Argument | None:
        """Resolve the nominal headed by ``token`` to a canonical entity.

        Args:
            token: the head token of the nominal.
            allow_literal: accept a LITERAL endpoint such as the object of ``focuses_on``.
            literal: the nominal *is* the literal, even though the coreference stage did
                not resolve it -- used for ``graph-based transaction analysis``, which the
                registry declares and which no pronoun or alias can ever point at.

        Returns ``None`` when the nominal is not an entity, which is the normal path for
        "leads the entity resolution work": ``work`` is a real noun chunk and a real
        object, but it is not in the registry, so no triple is emitted.
        """
        chunk = self.noun_chunk_of(token)
        if chunk is None:
            return None
        if literal:
            return self._literal_argument(chunk)
        return self.argument_for_span(chunk.start_char, chunk.end_char, allow_literal=allow_literal)

    def _literal_argument(self, chunk: Span) -> Argument | None:
        """Resolve a declared literal by its surface text."""
        if self.registry is None:
            return None
        text = chunk.text.strip()
        literal_id = self.registry.resolve_literal(text)
        if literal_id is None:
            # Longest-prefix match, so "the graph-based transaction analysis" and
            # "graph-based transaction analysis" reach the same node.
            literal_id = self.registry.resolve_literal(self._strip_determiner(text))
        if literal_id is None:
            return None
        return Argument(
            entity_id=literal_id,
            label=self.registry.label_of(literal_id),
            surface=text,
            entity_type=EntityType.LITERAL,
            was_coref_resolved=False,
        )

    @staticmethod
    def _strip_determiner(text: str) -> str:
        for article in ("the ", "a ", "an "):
            if text.lower().startswith(article):
                return text[len(article):]
        return text

    def literal_argument_for_token(self, token: Token) -> Argument | None:
        """Resolve a declared literal from a token's noun chunk."""
        chunk = self.noun_chunk_of(token)
        if chunk is None:
            return None
        return self._literal_argument(chunk)

    def argument_for_span(
        self, start_char: int, end_char: int, *, allow_literal: bool = False
    ) -> Argument | None:
        """Resolve a char span to a canonical entity via the coreference stage.

        Returns ``None`` when the span is not a resolved entity. That is the normal path
        for a real noun phrase that simply is not in the registry, such as "the entity
        resolution work" -- and it is the reason "She leads the entity resolution work"
        emits no triple while "She leads Project Aurora" does.
        """
        resolution = self.resolution_for(start_char, end_char)
        if resolution is None or not resolution.is_resolved or resolution.entity_id is None:
            return None
        entity_type = self._entity_type(resolution.entity_id)
        if entity_type is None:
            return None
        if entity_type is EntityType.LITERAL and not allow_literal:
            return None
        return Argument(
            entity_id=resolution.entity_id,
            label=resolution.resolved_label or resolution.entity_id,
            # Provenance is the resolved *mention*, not the enclosing noun chunk. The
            # chunk for "the data scientist leading Project Aurora" would otherwise be
            # recorded as the object surface of the leads triple.
            surface=resolution.mention or self.doc_text[start_char:end_char],
            entity_type=entity_type,
            # `is_anaphoric`, not the resolution layer: a definite description resolved
            # straight from the declared lexicon still *referred* to an entity, so the
            # triple is coreference-dependent (0.90) even though FastCoref was not used.
            was_coref_resolved=resolution.is_anaphoric,
            mention=None,
            resolution=resolution,
        )

    def _entity_type(self, entity_id: str) -> EntityType | None:
        if self.registry is not None:
            return self.registry.type_of(entity_id)
        # Fallback for contexts built without a registry (unit tests).
        for mention in self.processed.mentions:
            if mention.entity_id == entity_id and mention.entity_type is not None:
                return mention.entity_type
        return None

    # -- guards -------------------------------------------------------------

    def is_under_reporting_verb(self, token: Token) -> bool:
        """True when ``token`` sits inside a clause introduced by a reporting verb.

        Walks ancestors by index: a fresh wrapper is created on every ``.head`` access, so
        ``current.head is not current`` never becomes false and the walk would never
        terminate.
        """
        current = token
        seen: set[int] = set()
        while current.i not in seen:
            seen.add(current.i)
            if current.lemma_.casefold() in REPORTING_VERBS:
                return True
            parent = current.head
            if parent.i == current.i:  # reached the ROOT token
                break
            current = parent
        # Also check the other direction: "said [that the extension will reuse ...]"
        # makes the reporting verb the parent of the clause.
        for child in token.children:
            if child.dep_ in {"ccomp", "advcl", "acl"} and child.lemma_.casefold() in REPORTING_VERBS:
                return True
        return False

    def skip(
        self,
        rule_id: str,
        relation: str,
        reason: str,
    ) -> SkippedCandidate:
        """Record a considered-but-not-emitted relation (rule E2)."""
        return SkippedCandidate(
            rule_id=rule_id,
            relation=relation,
            reason=reason,
            document_id=self.processed.document.document_id,
            sentence_index=self.sentence_index,
            sentence=self.text,
        )


def find_predicates(ctx: RuleContext, lemmas: Iterable[str]) -> list[Token]:
    """Every token in the sentence whose lemma is in ``lemmas`` and pos is VERB.

    Finding by lemma rather than by dependency label is required by parser facts P2
    (``partnered`` is ``acl``) and P3 (``leading`` is ``amod``).
    """
    wanted = {lemma.casefold() for lemma in lemmas}
    return [
        token for token in ctx.sentence
        if token.pos_ in {"VERB", "AUX"} and token.lemma_.casefold() in wanted
    ]


def confidence_for(
    *,
    subject: Argument,
    obj: Argument,
    inferred: bool = False,
) -> ConfidenceBand:
    """Pick the deterministic confidence band (specification section 12).

    1.00 both endpoints are named entities stated outright.
    0.90 at least one endpoint arrived through coreference or normalisation.
    0.80 the relation itself is a semantic inference rather than a stated one.

    These are heuristic labels, not probabilities (charter invariant I7).
    """
    if inferred:
        return ConfidenceBand.INFERRED
    if subject.was_coref_resolved or obj.was_coref_resolved:
        return ConfidenceBand.COREF_OR_NORMALISED
    return ConfidenceBand.DIRECT


def build_candidate(
    ctx: RuleContext,
    *,
    rule_id: str,
    relation: RelationName,
    subject: Argument,
    obj: Argument,
    inferred: bool = False,
    context_entity: str = "",
    notes: str = "",
) -> RelationCandidate:
    """Assemble a :class:`RelationCandidate` with full provenance."""
    return RelationCandidate(
        relation=relation,
        subject=subject.label,
        object=obj.label,
        subject_original=subject.surface,
        object_original=obj.surface,
        document_id=ctx.processed.document.document_id,
        sentence_index=ctx.sentence_index,
        sentence=ctx.text,
        rule_id=rule_id,
        confidence=confidence_for(subject=subject, obj=obj, inferred=inferred),
        inferred=inferred,
        subject_id=subject.entity_id,
        object_id=obj.entity_id,
        subject_type=subject.entity_type,
        object_type=obj.entity_type,
        coref_resolved=subject.was_coref_resolved or obj.was_coref_resolved,
        context_entity=context_entity,
        notes=notes,
    )


def is_generic_nominal(text: str) -> bool:
    """True when a noun phrase is a common noun that must not become an endpoint."""
    from ..entity_resolver import normalise_surface

    head = normalise_surface(text).split()[-1] if normalise_surface(text) else ""
    return head in _NOT_A_RELATION_ENDPOINT


__all__ = [
    "Argument",
    "RuleContext",
    "REPORTING_VERBS",
    "build_candidate",
    "confidence_for",
    "find_predicates",
    "is_generic_nominal",
    "Sequence",
]
