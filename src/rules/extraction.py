"""Stage 5 -- deterministic relation extraction (specification sections 10 and 11).

Each rule is a named, testable unit that answers one question: *given this dependency
pattern, which canonical entities are related, and by what relation?* Rules never invent
an endpoint. A nominal that the coreference stage did not resolve to a canonical entity
produces a recorded skip, never a guess.

Two rules deserve their own note because they do not do what their verb says, and the
specification is explicit that they must not:

``R_INTEGRATED_INTO``
    "Aether Analytics plans to integrate Aurora into OrionEdge" must yield
    ``(Project Aurora, integrated_into, OrionEdge)``. The grammatical subject is the
    company, but the company is the *evidence context*, not the relation's subject.
    Storing ``(Aether Analytics, integrates, Aurora)`` would misplace the fact.

``R_CUSTOMER_OF``
    "Helios Bank renewed its contract with Aether Analytics" never states that Helios is
    a customer. It is inferred from renewing a contract, so it is emitted with
    ``inferred=True`` and the lower ``0.80`` band (specification section 11).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..logging_utils import get_logger
from ..models import RelationCandidate, RelationName, SkippedCandidate
from .base import (
    Argument,
    RuleContext,
    build_candidate,
    find_predicates,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.tokens import Token

    from ..entity_resolver import EntityRegistry

logger = get_logger(__name__)


class RelationRule(Protocol):
    """A single extraction rule.

    A rule never raises for "no relation here"; it returns nothing. Reporting a genuine
    defect is the extractor's job (rule E3).
    """

    rule_id: str
    relation: RelationName
    optional: bool

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        ...


#: Dependency labels under which a verb modifies a noun rather than owning an object
#: child. A reduced relative's "object" is the nominal it attaches to (parser fact P3).
_REDUCED_RELATIVE_DEPS = frozenset({"amod", "acl", "acl:relcl", "nmod"})


# ---------------------------------------------------------------------------
# Simple transitive verbs: S -> V -> O
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitiveVerbRule:
    """``<subject> <verb> <object>`` with a direct object.

    Covers develops, leads, improves, uses, focuses_on, works_on, collaborates_with and
    deployed_at. ``verb_lemmas`` is matched by lemma and ``prep_lemmas`` optionally
    redirects the object to a prepositional phrase.
    """

    rule_id: str
    relation: RelationName
    verb_lemmas: frozenset[str]
    prep_lemmas: frozenset[str] = frozenset()
    optional: bool = False
    allow_literal_object: bool = False
    allow_prep_object: bool = True

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for verb in find_predicates(ctx, self.verb_lemmas):
            if ctx.is_under_reporting_verb(verb):
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' sits inside a reporting verb clause; "
                               f"the sentence reports it rather than asserting it")
                continue

            subject_arg = self._subject(ctx, verb)
            if subject_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"no resolved subject for '{verb.text}'")
                continue

            object_arg, object_note = self._object(ctx, verb)
            if object_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"object of '{verb.text}' is not a canonical entity "
                               f"({object_note})")
                continue

            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=subject_arg,
                obj=object_arg,
                notes=object_note,
            )

    def _subject(self, ctx: RuleContext, verb: Token) -> Argument | None:
        for token in ctx.subject_tokens(verb):
            arg = ctx.argument_for_token(token)
            if arg is not None:
                return arg
        return None

    def _object(self, ctx: RuleContext, verb: Token) -> tuple[Argument | None, str]:
        if self.prep_lemmas and self.allow_prep_object and (
            pair := ctx.preposition(verb, self.prep_lemmas)
        ):
            prep, pobj = pair
            arg = ctx.argument_for_token(
                pobj, allow_literal=self.allow_literal_object
            )
            if arg is None and self.allow_literal_object:
                arg = ctx.literal_argument_for_token(pobj)
            if arg is not None:
                return arg, f"object via '{prep.text}' phrase"
        dobj = ctx.direct_object(verb)
        if dobj is None:
            # A reduced relative has no object *child*: in "the data scientist leading
            # Project Aurora" the verb is an `amod` modifier of Project Aurora, so the
            # nominal it governs is its own head (parser fact P3).
            dobj = verb.head if verb.head.i != verb.i and verb.dep_ in _REDUCED_RELATIVE_DEPS else None
        if dobj is not None:
            arg = ctx.argument_for_token(dobj, allow_literal=self.allow_literal_object)
            if arg is None and self.allow_literal_object:
                arg = ctx.literal_argument_for_token(dobj)
            if arg is not None:
                return arg, "object via direct object"
            return None, f"'{ctx.nominal_text(dobj)}' is not a canonical entity"
        return None, f"'{verb.text}' has no object"


@dataclass(frozen=True, slots=True)
class PassiveAgentRule:
    """``<object> was <verb> by <agent>`` -> ``(agent, verb, object)``.

    Handles "Aether Analytics was founded by Dr. Mira Sen" and yields
    ``(Dr. Mira Sen, founded, Aether Analytics)``: the grammatical subject is the thing
    founded, so the canonical direction is inverted.

    The agent preposition carries dep ``agent``, not ``prep`` (parser fact P1), which is
    why :meth:`RuleContext.preposition` accepts both.
    """

    rule_id: str
    relation: RelationName
    verb_lemmas: frozenset[str]
    agent_preps: frozenset[str] = frozenset({"by"})
    optional: bool = False

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for verb in find_predicates(ctx, self.verb_lemmas):
            if ctx.is_under_reporting_verb(verb):
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' sits inside a reporting verb clause")
                continue

            pair = ctx.preposition(verb, self.agent_preps)
            patient = None
            for token in verb.children:
                if token.dep_ == "nsubjpass":
                    patient = token
                    break
            if patient is None:
                for token in verb.children:
                    if token.dep_ in {"nsubj", "csubj"}:
                        patient = token
                        break

            if patient is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' has no patient subject")
                continue

            patient_arg = ctx.argument_for_token(patient)
            if patient_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"patient of '{verb.text}' is not a canonical entity")
                continue

            if pair is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' has no agent preposition; cannot invert")
                continue

            _, agent_pobj = pair
            agent_arg = ctx.argument_for_token(agent_pobj)
            if agent_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"agent of '{verb.text}' is not a canonical entity")
                continue

            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=agent_arg,
                obj=patient_arg,
                notes="passive construction; canonical direction inverted so the "
                      "relation reads agent -> verb -> patient",
            )


@dataclass(frozen=True, slots=True)
class ParticipialPartnershipRule:
    """``<ORG> partnered with <ORG>`` where ``partnered`` is an ``acl``.

    Parser fact P2: in "Aether Analytics partnered with Quantum Forge" the verb is
    ``dep=acl`` whose head is the ROOT noun, so the subject is the verb's head rather
    than a ``nsubj`` child.
    """

    rule_id: str = "R_PARTNERS_WITH"
    relation: RelationName = RelationName.PARTNERS_WITH
    verb_lemmas: frozenset[str] = frozenset({"partner"})
    prep_lemmas: frozenset[str] = frozenset({"with"})
    optional: bool = False

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for verb in find_predicates(ctx, self.verb_lemmas):
            if ctx.is_under_reporting_verb(verb):
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' sits inside a reporting verb clause")
                continue

            pair = ctx.preposition(verb, self.prep_lemmas)
            if pair is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' has no '{list(self.prep_lemmas)[0]}' phrase")
                continue
            _, pobj = pair

            subject_arg = self._host_or_subject(ctx, verb)
            if subject_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"host of participial '{verb.text}' is not a canonical entity")
                continue
            object_arg = ctx.argument_for_token(pobj)
            if object_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"partner of '{verb.text}' is not a canonical entity")
                continue

            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=subject_arg,
                obj=object_arg,
                notes="participial (acl) predicate; subject taken from the verb's head",
            )

    @staticmethod
    def _host_or_subject(ctx: RuleContext, verb: Token) -> Argument | None:
        if verb.dep_ in {"acl", "acl:relcl"}:
            return ctx.argument_for_token(verb.head)
        for token in ctx.subject_tokens(verb):
            if (arg := ctx.argument_for_token(token)) is not None:
                return arg
        return None


@dataclass(frozen=True, slots=True)
class ModalXcompIntegrationRule:
    """``<ORG> plans to integrate <X> into <Y>`` -> ``(X, integrated_into, Y)``.

    Parser fact P6: the real predicate sits in an ``xcomp`` under the modal, so the
    subject of the sentence is *not* the relation's subject. The organisation stays as
    ``context_entity`` because it is where the statement was made (specification section
    11).
    """

    rule_id: str = "R_INTEGRATED_INTO"
    relation: RelationName = RelationName.INTEGRATED_INTO
    verb_lemmas: frozenset[str] = frozenset({"integrate"})
    prep_lemmas: frozenset[str] = frozenset({"into"})
    optional: bool = False

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for verb in find_predicates(ctx, self.verb_lemmas):
            if ctx.is_under_reporting_verb(verb):
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' sits inside a reporting verb clause")
                continue

            pair = ctx.preposition(verb, self.prep_lemmas)
            dobj = ctx.direct_object(verb)
            if pair is None or dobj is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' lacks a dobj + 'into' phrase")
                continue
            _, pobj = pair

            integrated = ctx.argument_for_token(dobj)
            if integrated is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{ctx.nominal_text(dobj)}' is not a canonical entity")
                continue
            host = ctx.argument_for_token(pobj)
            if host is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{ctx.nominal_text(pobj)}' is not a canonical entity")
                continue

            context = self._speaker_context(ctx, verb)
            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=integrated,
                obj=host,
                context_entity=context or "",
                notes="integration target is the relation's subject; the organisation "
                      "named in the sentence is the evidence context, not the subject",
            )

    @staticmethod
    def _speaker_context(ctx: RuleContext, verb: Token) -> str:
        """Find the organisation whose statement this is, by walking up the tree.

        The walk stops when a token is its own head. That test compares token *indices*:
        spaCy returns a new Python wrapper for every ``.head`` access, so an identity
        test would never stop and this loop would spin forever.
        """
        walker = verb.head
        while walker is not None and walker.head.i != walker.i:
            for token in ctx.subject_tokens(walker):
                arg = ctx.argument_for_token(token)
                if arg is not None:
                    return arg.label
            walker = walker.head
        return ""


@dataclass(frozen=True, slots=True)
class ContractRenewalInferenceRule:
    """``<ORG> renewed its contract with <ORG>`` -> ``(<ORG>, customer_of, <ORG>)``.

    This relation is **not stated**. Renewing a contract with a vendor is evidence of a
    customer relationship, so the candidate is marked ``inferred=True`` and receives the
    ``0.80`` band. The specification requires exactly this distinction and forbids
    presenting an inference as though the text said it.
    """

    rule_id: str = "R_CUSTOMER_OF"
    relation: RelationName = RelationName.CUSTOMER_OF
    verb_lemmas: frozenset[str] = frozenset({"renew"})
    prep_lemmas: frozenset[str] = frozenset({"with"})
    object_nouns: frozenset[str] = frozenset({"contract", "agreement", "subscription"})
    optional: bool = False

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for verb in find_predicates(ctx, self.verb_lemmas):
            if ctx.is_under_reporting_verb(verb):
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' sits inside a reporting verb clause")
                continue

            subject_arg = next(
                (arg for t in ctx.subject_tokens(verb)
                 if (arg := ctx.argument_for_token(t)) is not None),
                None,
            )
            if subject_arg is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"no resolved subject for '{verb.text}'")
                continue

            # Require the stated object to be a commercial contract, so "renewed its
            # passport with the embassy" cannot become a customer relation.
            direct = ctx.direct_object(verb)
            if direct is None or direct.lemma_.casefold() not in self.object_nouns:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"object of '{verb.text}' is not a contract noun; "
                               f"no customer relation is implied")
                continue

            pair = ctx.preposition(verb, self.prep_lemmas)
            if pair is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'{verb.text}' has no 'with' phrase")
                continue
            _, pobj = pair
            vendor = ctx.argument_for_token(pobj)
            if vendor is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"'with' partner of '{verb.text}' is not a canonical entity")
                continue

            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=subject_arg,
                obj=vendor,
                inferred=True,
                notes=f"INFERRED: '{verb.text} ... contract {list(self.prep_lemmas)[0]} ...' "
                      f"is evidence of a customer relationship, not a statement of one",
            )


@dataclass(frozen=True, slots=True)
class PossessiveContainmentRule:
    """``<OWNER>'s <COMPONENT>`` -> ``(<OWNER>, contains, <COMPONENT>)``.

    Optional relation. A possessive is a reliable structural signal here, so the rule is
    enabled by default but still gated by ``enable_optional_relations``.
    """

    rule_id: str = "R_CONTAINS_POSSESSIVE"
    relation: RelationName = RelationName.CONTAINS
    optional: bool = True

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for token in ctx.sentence:
            if token.dep_ != "poss":
                continue
            # Own span first: "Aurora's graph matching module" is one noun chunk, but the
            # possessor is "Aurora" alone and the chunk would leak into the provenance.
            owner = ctx.argument_for_own_token(token)
            if owner is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"possessor '{token.text}' is not a canonical entity")
                continue
            possessed = ctx.argument_for_token(token.head)
            if possessed is None:
                yield ctx.skip(self.rule_id, self.relation.value,
                               f"possessed '{token.head.text}' is not a canonical entity")
                continue
            if owner.entity_id == possessed.entity_id:
                continue
            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=owner,
                obj=possessed,
                notes=f"possessive construction '{token.text} {token.head.text}'",
            )


@dataclass(frozen=True, slots=True)
class PrepositionalContainmentRule:
    """``<X> <improve verb> <Y> inside <Z>`` -> ``(<Z>, contains, <Y>)``.

    The preposition's object is the *container*, so the direction is reversed relative to
    the sentence: "the Entity Resolution Engine **inside** Project Aurora" gives
    ``(Project Aurora, contains, Entity Resolution Engine)``.
    """

    rule_id: str = "R_CONTAINS_INSIDE"
    relation: RelationName = RelationName.CONTAINS
    container_preps: frozenset[str] = frozenset({"inside", "within"})
    optional: bool = True

    def apply(self, ctx: RuleContext) -> Iterator[RelationCandidate | SkippedCandidate]:
        for prep, pobj in _all_preps(ctx, self.container_preps):
            container = ctx.argument_for_token(pobj)
            if container is None:
                continue
            host = prep.head
            inner = ctx.direct_object(host) or (host.head if host.dep_ == "prep" else None)
            if inner is None:
                continue
            contained = ctx.argument_for_token(inner)
            if contained is None or contained.entity_id == container.entity_id:
                continue
            yield build_candidate(
                ctx,
                rule_id=self.rule_id,
                relation=self.relation,
                subject=container,
                obj=contained,
                notes=f"'{prep.text}' phrase reverses the direction: the "
                      f"preposition's object is the container",
            )


def _all_preps(ctx: RuleContext, preps: Iterable[str]) -> list[tuple[Token, Token]]:
    """Every ``(prep, pobj)`` pair in the sentence, not just under one verb."""
    wanted = {p.casefold() for p in preps}
    found: list[tuple[Token, Token]] = []
    for token in ctx.sentence:
        if token.dep_ in {"prep", "agent"} and token.lower_ in wanted:
            pobj = RuleContext._object_of_prep(token)
            if pobj is not None:
                found.append((token, pobj))
    return found


# ---------------------------------------------------------------------------
# Rule inventory
# ---------------------------------------------------------------------------

#: Every rule, in the order the extractor applies them. Order matters only for
#: duplicate detection; candidates are de-duplicated by ``(subject, relation, object)``.
RULES: tuple[RelationRule, ...] = (
    PassiveAgentRule(
        rule_id="R_FOUNDED_PASSIVE",
        relation=RelationName.FOUNDED,
        verb_lemmas=frozenset({"found", "establish"}),
    ),
    TransitiveVerbRule(
        rule_id="R_DEVELOPS",
        relation=RelationName.DEVELOPS,
        verb_lemmas=frozenset({"develop", "build", "create"}),
    ),
    TransitiveVerbRule(
        rule_id="R_LEADS",
        relation=RelationName.LEADS,
        verb_lemmas=frozenset({"lead", "head"}),
    ),
    TransitiveVerbRule(
        rule_id="R_FOCUSES_ON",
        relation=RelationName.FOCUSES_ON,
        verb_lemmas=frozenset({"focus", "concentrate"}),
        prep_lemmas=frozenset({"on", "upon"}),
        allow_literal_object=True,
    ),
    TransitiveVerbRule(
        rule_id="R_JOINED_EMPLOYMENT",
        relation=RelationName.WORKS_AT,
        # "joined" is normalised to works_at: the graph records the current employment
        # relationship, which the specification explicitly sanctions.
        verb_lemmas=frozenset({"join"}),
    ),
    TransitiveVerbRule(
        rule_id="R_WORKS_ON",
        relation=RelationName.WORKS_ON,
        verb_lemmas=frozenset({"work"}),
        prep_lemmas=frozenset({"on", "at"}),
    ),
    TransitiveVerbRule(
        rule_id="R_COLLABORATES_WITH",
        relation=RelationName.COLLABORATES_WITH,
        verb_lemmas=frozenset({"collaborate"}),
        prep_lemmas=frozenset({"with"}),
    ),
    ParticipialPartnershipRule(),
    TransitiveVerbRule(
        rule_id="R_DEPLOYED_AT",
        relation=RelationName.DEPLOYED_AT,
        verb_lemmas=frozenset({"deploy", "install"}),
        prep_lemmas=frozenset({"at", "in"}),
    ),
    TransitiveVerbRule(
        rule_id="R_USES",
        relation=RelationName.USES,
        verb_lemmas=frozenset({"use", "utilise", "leverage"}),
    ),
    TransitiveVerbRule(
        rule_id="R_IMPROVES",
        relation=RelationName.IMPROVES,
        verb_lemmas=frozenset({"improve", "enhance"}),
    ),
    ModalXcompIntegrationRule(),
    ContractRenewalInferenceRule(),
    PrepositionalContainmentRule(),
    PossessiveContainmentRule(),
)


class RelationExtractor:
    """Applies every enabled rule to every sentence of every document."""

    def __init__(self, enable_optional_relations: bool = True) -> None:
        self._enable_optional = enable_optional_relations
        self._rules = tuple(r for r in RULES if enable_optional_relations or not r.optional)
        skipped_optional = [r.rule_id for r in RULES if r.optional and not enable_optional_relations]
        if skipped_optional:
            logger.info("Optional relations disabled; skipping rules %s",
                        ", ".join(skipped_optional))

    @property
    def rules(self) -> tuple[RelationRule, ...]:
        return self._rules

    def extract(
        self,
        processed_documents: Iterable[tuple[Any, Mapping[tuple[int, int], Any]]],
        registry: EntityRegistry | None = None,
    ) -> tuple[tuple[RelationCandidate, ...], tuple[SkippedCandidate, ...]]:
        """Run every rule over every sentence.

        Args:
            processed_documents: ``(ProcessedDocument, {span: MentionResolution})`` pairs.
            registry: canonical registry, used for entity types and LITERAL endpoints.

        Returns:
            ``(candidates, skipped)``. Skipped candidates are returned rather than
            logged away: showing what the extractor declined to emit is how a reader can
            tell "no rule fired" from "a rule fired and refused".
        """
        candidates: list[RelationCandidate] = []
        skipped: list[SkippedCandidate] = []

        for processed, resolutions in processed_documents:
            for sentence_index, sentence in enumerate(processed.doc.sents):
                ctx = RuleContext(
                    processed=processed,
                    sentence=sentence,
                    sentence_index=sentence_index,
                    resolutions=resolutions,
                    doc_text=processed.document.text,
                    registry=registry,
                )
                for rule in self._rules:
                    for outcome in rule.apply(ctx):
                        if isinstance(outcome, RelationCandidate):
                            candidates.append(outcome)
                        else:
                            skipped.append(outcome)

        logger.info(
            "Relation extraction: %d candidate(s), %d skipped",
            len(candidates), len(skipped),
        )
        return tuple(candidates), tuple(skipped)


__all__ = [
    "RelationExtractor",
    "RULES",
    "TransitiveVerbRule",
    "PassiveAgentRule",
    "ParticipialPartnershipRule",
    "ModalXcompIntegrationRule",
    "ContractRenewalInferenceRule",
    "PrepositionalContainmentRule",
    "PossessiveContainmentRule",
    "RelationRule",
]
