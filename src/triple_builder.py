"""Stage 6 -- triple construction.

Turns :class:`~src.models.RelationCandidate` objects into
:class:`~src.models.Triple` objects, enforcing the constraints that make a triple
trustworthy:

* both endpoints must resolve to real canonical ids;
* a subject and an object may not be the same entity;
* duplicates collapse, and the surviving triple records how many sentences supported it;
* provenance survives intact.

Every rejection is counted by reason so the pipeline report can show *why* a candidate did
not become a triple, rather than silently dropping it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import NamedTuple

from .entity_resolver import EntityRegistry
from .ids import make_triple_id
from .logging_utils import get_logger
from .models import RelationCandidate, Triple

logger = get_logger(__name__)


class RejectionCounts(NamedTuple):
    """Why candidates did not become triples. Reported, never hidden."""

    unresolved_endpoint: int = 0
    self_loop: int = 0
    literal_endpoint: int = 0


class TripleBuilder:
    """Builds validated, de-duplicated triples."""

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    def build(
        self, candidates: Iterable[RelationCandidate]
    ) -> tuple[tuple[Triple, ...], RejectionCounts]:
        """Validate candidates and collapse duplicates.

        Duplicate collapse keeps the *first* occurrence in document order and increments
        ``duplicate_count``, so re-running the pipeline on unchanged documents produces
        byte-identical artifacts.
        """
        accepted: dict[tuple[str, str, str, str], Triple] = {}
        rejections = {"unresolved_endpoint": 0, "self_loop": 0, "literal_endpoint": 0}

        for candidate in candidates:
            reason = self._rejection_reason(candidate)
            if reason:
                rejections[reason] += 1
                logger.debug(
                    "Rejected %s candidate %s --%s-> %s: %s",
                    candidate.rule_id, candidate.subject, candidate.relation.value,
                    candidate.object, reason.replace("_", " "),
                )
                continue

            triple = self._to_triple(candidate)
            key = (triple.subject_id, triple.relation.value, triple.object_id, triple.document_id)
            existing = accepted.get(key)
            accepted[key] = triple if existing is None else replace(
                existing, duplicate_count=existing.duplicate_count + 1
            )

        triples = tuple(accepted[key] for key in sorted(accepted))
        counts = RejectionCounts(**rejections)
        logger.info(
            "Built %d triple(s); rejected %d unresolved endpoint(s), %d self-loop(s)",
            len(triples), counts.unresolved_endpoint, counts.self_loop,
        )
        return triples, counts

    def _rejection_reason(self, candidate: RelationCandidate) -> str:
        if not candidate.endpoints_resolved:
            return "unresolved_endpoint"
        if candidate.subject_id == candidate.object_id:
            return "self_loop"
        # A LITERAL endpoint is legitimate for focuses_on but not for these relations:
        # "the project focuses on graph-based transaction analysis" is meaningful, while
        # "(something) improves (a literal)" is not something the corpus supports.
        if candidate.object_id.startswith("literal:") and candidate.relation.value != "focuses_on":
            return "literal_endpoint"
        return ""

    def _to_triple(self, candidate: RelationCandidate) -> Triple:
        return Triple(
            triple_id=make_triple_id(
                candidate.subject_id, candidate.relation.value,
                candidate.object_id, candidate.document_id,
            ),
            subject=self._registry.label_of(candidate.subject_id),
            subject_id=candidate.subject_id,
            relation=candidate.relation,
            object=self._registry.label_of(candidate.object_id),
            object_id=candidate.object_id,
            document_id=candidate.document_id,
            sentence_index=candidate.sentence_index,
            sentence=candidate.sentence,
            original_subject=candidate.subject_original,
            original_object=candidate.object_original,
            coref_resolved=candidate.coref_resolved,
            inferred=candidate.inferred,
            confidence=candidate.confidence,
            rule_id=candidate.rule_id,
            context_entity=candidate.context_entity,
        )


__all__ = ["TripleBuilder", "RejectionCounts"]
