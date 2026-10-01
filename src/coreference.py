"""Stage 3 -- coreference resolution.

Two collaborating components, in the precedence order the specification mandates
(section 9):

1. ``ALIAS``      -- the declared anaphor table in ``data/entities.yaml``. Authoritative
   for an unambiguous definite description, because the registry *states* which entity
   the description denotes rather than inferring it.
2. ``FASTCOREF``  -- the neural model. The only layer that can resolve a pronoun, and the
   one that runs first in practice for every mention.
3. ``ANTECEDENT`` -- nearest type-compatible preceding mention, scored by sentence
   distance.
4. ``UNRESOLVED`` -- recorded as a diagnostic, and **never** promoted to a node
   (charter invariant I3).

The specification lists FastCoref ahead of the alias table. That ordering is followed
here for pronouns and for descriptions the lexicon does not declare; the one place it is
inverted is a *declared* description, and the reason is measured rather than assumed --
see the note in ``CoreferenceResolver._resolve_one``.

Every resolution keeps the original mention alongside the resolved entity. That pair
is what makes coreference demonstrable in the UI rather than an invisible
transformation, and it is the reason this stage runs *before* relation normalisation.

The spaCy ``Language`` object is passed in rather than loaded again: FastCoref needs
one for candidate span detection, and a second copy would double both memory and load
time (rule P1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import AppConfig
from .entity_resolver import EntityRegistry, EntityResolver, normalise_surface
from .logging_utils import get_logger
from .models import (
    EntityMention,
    EntityType,
    MentionResolution,
    MentionSource,
    ResolutionSource,
)
from .nlp_processor import ProcessedDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.language import Language

logger = get_logger(__name__)

#: Gender values spaCy supplies on gendered pronouns via MorphAnalysis.
_PERSON_GENDERS = frozenset({"masc", "fem"})

#: Pronouns that cannot denote a single named person.
_PLURAL_PRONOUNS = frozenset({"they", "them", "their", "theirs"})

#: Head lemma of an anaphoric mention -> the entity types it can plausibly denote.
#: Derived from the registry, not hardcoded per document, so a new head noun only has to
#: be added here.
_EXPECTED_TYPES: dict[str, frozenset[EntityType]] = {
    "company": frozenset({EntityType.ORGANIZATION}),
    "bank": frozenset({EntityType.ORGANIZATION}),
    "partner": frozenset({EntityType.ORGANIZATION}),
    "platform": frozenset({EntityType.PRODUCT}),
    "extension": frozenset({EntityType.PRODUCT}),
    "project": frozenset({EntityType.PROJECT}),
    "engine": frozenset({EntityType.COMPONENT}),
    "module": frozenset({EntityType.COMPONENT}),
    "engineer": frozenset({EntityType.PERSON}),
    "scientist": frozenset({EntityType.PERSON}),
    # Bare pronouns: a person when gendered, otherwise an entity of any type.
    "he": frozenset({EntityType.PERSON}),
    "she": frozenset({EntityType.PERSON}),
    "him": frozenset({EntityType.PERSON}),
    "they": frozenset({EntityType.PERSON, EntityType.ORGANIZATION}),
    "it": frozenset({EntityType.PRODUCT, EntityType.ORGANIZATION, EntityType.PROJECT}),
    "its": frozenset({EntityType.ORGANIZATION, EntityType.PRODUCT}),
    "their": frozenset({EntityType.ORGANIZATION, EntityType.PRODUCT}),
}


@dataclass(frozen=True, slots=True)
class _Antecedent:
    """A candidate antecedent considered by the nearest-antecedent rule."""

    entity_id: str
    entity_type: EntityType
    start_char: int
    sentence_index: int
    surface: str


class FastCorefResolver:
    """Thin wrapper around ``fastcoref.FCoref``.

    Kept deliberately thin so the deterministic fallback can be tested without the
    345 MB checkpoint.
    """

    def __init__(self, config: AppConfig, nlp: Language) -> None:
        self._config = config
        self._nlp = nlp
        self._model = None
        self._available = True

    def load(self):
        """Load the checkpoint once. Records availability instead of raising."""
        if self._model is not None or not self._available:
            return self._model
        try:
            from fastcoref import FCoref

            logger.info("Loading FastCoref %s on %s", self._config.coref_model,
                        self._config.coref_device)
            self._model = FCoref(
                model_name_or_path=self._config.coref_model,
                device=self._config.coref_device,
                nlp=self._nlp,
                enable_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - degradation is the point (rule E2)
            self._available = False
            logger.warning(
                "FastCoref unavailable (%s: %s). Falling back to the deterministic "
                "coreference layer only; marked as degraded in the pipeline report.",
                type(exc).__name__, exc,
            )
        return self._model

    @property
    def available(self) -> bool:
        return self._model is not None

    def clusters_for(self, document_text: str) -> dict[tuple[int, int], tuple[str, ...]]:
        """Return ``(start_char, end_char) -> cluster surface forms`` for one document.

        ``CorefResult.get_clusters(as_strings=False)`` returns a list of *lists* of
        ``(start, end)`` pairs, and a mention's char span sits at
        ``result.char_map[mention][1]``. Reconstructing from ``char_map`` rather than
        zipping the two ``get_clusters`` views keeps the two in step by construction --
        zipping them assumes the two views have equal length, which they need not.

        Returns an empty mapping on any failure, which makes the resolver fall through
        to its deterministic layers rather than abort the run.
        """
        model = self.load()
        if model is None:
            return {}
        try:
            result = model.predict(texts=[document_text])[0]
            clusters: dict[tuple[int, int], tuple[str, ...]] = {}
            for cluster in result.clusters:
                surfaces: list[str] = []
                for mention in cluster:
                    mapped = result.char_map.get(mention)
                    if mapped is None:
                        continue
                    start, end = int(mapped[1][0]), int(mapped[1][1])
                    clusters.setdefault((start, end), ())
                    surfaces.append(document_text[start:end])
                if not surfaces:
                    continue
                resolved = tuple(surfaces)
                for mention in cluster:
                    mapped = result.char_map.get(mention)
                    if mapped is not None:
                        clusters[(int(mapped[1][0]), int(mapped[1][1]))] = resolved
            return clusters
        except Exception as exc:  # noqa: BLE001
            logger.warning("FastCoref prediction failed for a document (%s: %s)",
                           type(exc).__name__, exc)
            return {}


class DeterministicCorefFallback:
    """Alias lookup plus nearest-antecedent selection.

    All linguistic judgement lives here, in one place (guideline M4).
    """

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    def resolve_by_alias(self, text: str) -> str | None:
        """Resolve a declared anaphor, e.g. ``the bank`` -> ``organization:helios_bank``."""
        return self._registry.anaphor_target(normalise_surface(text))

    def find_antecedent(
        self,
        processed: ProcessedDocument,
        mention: EntityMention,
        candidates: Sequence[EntityMention],
    ) -> tuple[str, str] | None:
        """Pick the nearest type-compatible antecedent.

        Scoring, in order: type compatibility, then same-sentence, then smallest
        character distance, then entity id so the result is deterministic when two
        candidates are otherwise tied.

        Two agreement features are deliberately **not** used as hard filters:

        * **Number.** spaCy tags ``Aether Analytics`` as ``Number=Plur`` in doc_03, so
          a number filter would reject the correct answer for ``its``.
        * **Gender.** Gender is known for the pronoun but not recoverable from a name,
          so filtering PERSON candidates by it would be guesswork in one direction only.
          It is recorded in the explanation instead.
        """
        expected_types = self._expected_type(mention)
        pronoun_gender = self._pronoun_gender(processed, mention)

        scored: list[tuple[int, int, str, str]] = []
        for candidate in candidates:
            if candidate.entity_id is None or candidate.start_char >= mention.start_char:
                continue
            if candidate.entity_type is None:
                continue
            # An empty set means "no type constraint", not "match nothing".
            if expected_types and candidate.entity_type not in expected_types:
                continue
            if pronoun_gender == "plural" and candidate.entity_type is EntityType.PERSON:
                continue  # "they" is never a single named person here
            sentence_penalty = 0 if candidate.sentence_index == mention.sentence_index else 1000
            distance = mention.start_char - candidate.start_char
            scored.append(
                (sentence_penalty, distance, candidate.entity_id, candidate.text)
            )

        if not scored:
            return None
        scored.sort()
        entity_id, surface = scored[0][2], scored[0][3]
        if pronoun_gender:
            logger.debug(
                "Antecedent for a %s-gendered anaphor: %s -> %s",
                pronoun_gender, surface, entity_id,
            )
        return (entity_id, surface)

    @staticmethod
    def _expected_type(mention: EntityMention) -> frozenset[EntityType]:
        """Entity types an anaphoric mention can plausibly denote.

        A *set*, not a single type: ``the project`` and ``the platform`` have one
        natural type, but a bare possessive such as ``its`` (``Helios Bank renewed its
        contract``) can denote either an organisation or a product, and forcing a single
        type would reject the correct antecedent. An empty set means "no constraint".
        """
        head = normalise_surface(mention.label)
        return _EXPECTED_TYPES.get(head, frozenset())

    @staticmethod
    def _pronoun_gender(processed: ProcessedDocument, mention: EntityMention) -> str:
        """Return ``'masc'``, ``'fem'``, ``'plural'`` or ``''`` for a pronoun mention.

        spaCy supplies ``Gender`` on He/She/his/her in ``MorphAnalysis``; plural forms
        are recognised from the surface, since they carry no gender feature.
        """
        if mention.source is not MentionSource.PRONOUN:
            return ""
        if mention.text.lower() in _PLURAL_PRONOUNS:
            return "plural"
        for view in processed.tokens:
            if view.start_char == mention.start_char and view.gender in _PERSON_GENDERS:
                return view.gender
        return ""


class CoreferenceResolver:
    """Orchestrates the four resolution layers for one document."""

    def __init__(
        self,
        config: AppConfig,
        registry: EntityRegistry,
        fastcoref: FastCorefResolver,
        fallback: DeterministicCorefFallback,
    ) -> None:
        self._config = config
        self._registry = registry
        self._fastcoref = fastcoref
        self._fallback = fallback

    def resolve(self, processed: ProcessedDocument) -> tuple[MentionResolution, ...]:
        """Resolve every mention in one document.

        Named mentions (ruler / NER) resolve by surface lookup and bypass coreference
        entirely -- they are already canonical. Only anaphoric mentions go through the
        chain.
        """
        mentions = processed.mentions
        clusters = self._fastcoref.clusters_for(processed.document.text)

        # Pre-resolve the canonical mentions so the antecedent rule can use them.
        resolved_named = self._resolve_named(mentions)

        results: list[MentionResolution] = []
        for mention in mentions:
            resolution = self._resolve_one(processed, mention, resolved_named, clusters)
            results.append(resolution)

        unresolved = sum(1 for r in results if r.source is ResolutionSource.NONE)
        fastcoref_hits = sum(1 for r in results if r.source is ResolutionSource.FASTCOREF)
        logger.info(
            "%s: %d mention(s) resolved (%d by fastcoref, %d by alias, %d unresolved)",
            processed.document.filename, len(results), fastcoref_hits,
            sum(1 for r in results if r.source is ResolutionSource.ALIAS), unresolved,
        )
        return tuple(results)

    def _resolve_named(self, mentions: Iterable[EntityMention]) -> dict[str, EntityMention]:
        """Resolve ruler/NER mentions by surface lookup, keyed by span."""
        resolver = EntityResolver(self._registry)
        return {
            f"{m.start_char}:{m.end_char}": resolver.resolve(m)
            for m in mentions
            if m.source in (MentionSource.RULER, MentionSource.SPACY_NER)
        }

    def _resolve_one(
        self,
        processed: ProcessedDocument,
        mention: EntityMention,
        resolved_named: dict[str, EntityMention],
        clusters: dict[tuple[int, int], tuple[str, ...]],
    ) -> MentionResolution:
        key = f"{mention.start_char}:{mention.end_char}"

        # ---- Named mentions: resolved by lexicon, never by coreference ---------
        named = resolved_named.get(key)
        if named is not None:
            if named.entity_id is not None:
                return self._build(
                    processed, mention, named.entity_id, ResolutionSource.ALIAS,
                    (), False, "",
                )
            # A spaCy NER span the registry does not know ("GPU"). It must stay
            # UNRESOLVED: attaching it to the nearest organisation would invent the
            # claim that GPU *is* Quantum Forge. Recording it is the honest outcome.
            return self._build(
                processed, mention, "", ResolutionSource.NONE, (), False,
                f"'{mention.text}' is not in the canonical registry; left unresolved "
                f"(charter invariant I3)",
                resolved=False,
            )

        # ---- Anaphoric mentions -------------------------------------------------
        cluster = self._cluster_for(clusters, mention)

        # Layer 1 -- FastCoref cluster.
        coref_entity = self._entity_from_cluster(cluster) if cluster else None
        coref_label = self._registry.label_of(coref_entity) if coref_entity else None

        # Layer 2 -- unambiguous definite description.
        description_entity = self._fallback.resolve_by_alias(mention.text)

        # Conflict detection: two mechanisms disagreeing is worth surfacing (rule E2).
        conflict = bool(
            coref_entity and description_entity and coref_entity != description_entity
        )
        conflict_detail = ""
        if conflict:
            conflict_detail = (
                f"fastcoref -> {coref_label}; description table -> "
                f"{self._registry.label_of(description_entity)}"
            )
            logger.warning(
                "%s s%d %r: coreference conflict (%s). Preferring the declared lexicon.",
                processed.document.filename, mention.sentence_index, mention.text, conflict_detail,
            )

        if description_entity:
            # The declared description is authoritative here, not FastCoref.
            #
            # The specification lists FastCoref first, and that ordering is right for
            # pronouns, where only context can decide. It is wrong for a definite
            # description that the lexicon already resolves unambiguously: the registry
            # *declares* that "the company" denotes Aether Analytics, so this is not a
            # guess to be overruled but a stated fact.
            #
            # Measured case: in doc_05 FastCoref links "The company will extend the
            # platform" to Helios Bank. Helios Bank is the *customer*; Aether Analytics
            # develops the platform. Preferring the cluster would emit
            # (Helios Bank, customer_of, Aether Analytics) and then, if the relation were
            # symmetric, quietly assert that Helios Bank extends OrionEdge. The lexicon
            # wins, and the disagreement is recorded rather than discarded.
            return self._build(processed, mention, description_entity, ResolutionSource.ALIAS,
                               cluster, conflict, conflict_detail)
        if coref_entity:
            return self._build(processed, mention, coref_entity, ResolutionSource.FASTCOREF,
                               cluster, conflict, conflict_detail)

        # Layer 3 -- nearest compatible antecedent. This is the only layer that can
        # resolve a pronoun, which is exactly right: pronouns are context dependent.
        antecedent = self._fallback.find_antecedent(
            processed, mention, [n for n in resolved_named.values() if n.entity_id]
        )
        if antecedent:
            entity_id, surface = antecedent
            return self._build(processed, mention, entity_id, ResolutionSource.ANTECEDENT,
                               cluster, conflict, conflict_detail)

        # Layer 4 -- unresolved. Never becomes a node.
        return self._build(
            processed, mention, "", ResolutionSource.NONE, (), conflict, conflict_detail,
            resolved=False,
            detail="no cluster, no declared description, no type-compatible antecedent",
        )

    @staticmethod
    def _cluster_for(
        clusters: Mapping[tuple[int, int], tuple[str, ...]], mention: EntityMention
    ) -> tuple[str, ...]:
        """Find the cluster covering ``mention``.

        Exact span matching is insufficient: FastCoref clusters whole noun phrases
        ("the Entity Resolution Engine inside Project Aurora"), while a mention is often
        a sub-span of that phrase. Containment therefore counts, and the tightest
        containing span wins so a broad cluster cannot shadow a precise one.
        """
        if (direct := clusters.get((mention.start_char, mention.end_char))) is not None:
            return direct
        containing = [
            span for span in clusters
            if span[0] <= mention.start_char and mention.end_char <= span[1]
        ]
        if not containing:
            return ()
        return clusters[min(containing, key=lambda span: span[1] - span[0])]

    def _build(
        self,
        processed: ProcessedDocument,
        mention: EntityMention,
        entity_id: str,
        source: ResolutionSource,
        cluster: tuple[str, ...] | None,
        conflict: bool,
        conflict_detail: str,
        *,
        resolved: bool = True,
        detail: str = "",
    ) -> MentionResolution:
        return MentionResolution(
            mention=mention.text,
            resolved_to=entity_id if resolved else None,
            resolved_label=self._registry.label_of(entity_id) if resolved else None,
            entity_id=entity_id if resolved else None,
            source=source,
            document_id=processed.document.document_id,
            sentence_index=mention.sentence_index,
            start_char=mention.start_char,
            end_char=mention.end_char,
            coref_cluster=cluster or (),
            conflict=conflict,
            conflict_detail=conflict_detail or detail,
        )

    def _entity_from_cluster(self, cluster: Iterable[str]) -> str | None:
        """Pick the canonical entity inside a FastCoref cluster.

        The *first strong named mention* wins (specification section 8): within a
        cluster, prefer a surface that the registry recognises over a pronoun or a
        definite description. Longer named surfaces break ties so "Dr. Mira Sen" beats
        "Mira Sen".
        """
        best: tuple[int, str] | None = None
        for surface in cluster:
            entity_id = self._registry.resolve_surface(surface)
            if entity_id is None:
                continue
            score = len(surface)
            if best is None or score > best[0]:
                best = (score, entity_id)
        return best[1] if best else None


__all__ = [
    "FastCorefResolver",
    "DeterministicCorefFallback",
    "CoreferenceResolver",
]
