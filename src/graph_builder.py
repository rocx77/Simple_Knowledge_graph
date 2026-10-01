"""S7a -- build a ``networkx.MultiDiGraph`` from validated triples.

Design decisions worth stating, because both are deliberate departures from the obvious
implementation:

* ``MultiDiGraph``, not ``DiGraph``. Two different relations between the same pair of
  entities stay separate edges, each keeping its own provenance. Collapsing them would
  lose the document and sentence a relation came from (specification section 13).
* No inverse edges. ``collaborates_with`` and ``partners_with`` are stored once in the
  direction the text states. The query engine searches both directions rather than
  doubling the graph.

Nodes are created only for entities that participate in at least one triple. A declared
entity with no relations still appears in ``entities.json`` carrying ``has_edges: false``,
so nothing the registry knows about is silently dropped from the artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import networkx as nx

from .entity_resolver import EntityRegistry
from .models import MentionResolution, Triple


class GraphBuilder:
    """Turns triples and the mention inventory into a queryable graph."""

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    def build(
        self,
        triples: Sequence[Triple],
        resolutions: Iterable[MentionResolution] = (),
    ) -> nx.MultiDiGraph:
        """Return the graph.

        ``resolutions`` supplies mention counts and the document list per node; without
        them those attributes fall back to zero and empty, which is why the pipeline
        always passes them.
        """
        mentions = self._mention_counts(resolutions)

        graph = nx.MultiDiGraph()
        for triple in triples:
            for entity_id in (triple.subject_id, triple.object_id):
                if entity_id not in graph:
                    self._add_node(graph, entity_id, mentions)

            graph.add_edge(
                triple.subject_id,
                triple.object_id,
                key=triple.triple_id,
                relation=triple.relation.value,
                confidence=float(triple.confidence),
                coref_resolved=triple.coref_resolved,
                document_id=triple.document_id,
                sentence_id=triple.sentence_index,
                sentence=triple.sentence,
                inferred=triple.inferred,
                rule_id=triple.rule_id,
                original_subject=triple.original_subject,
                original_object=triple.original_object,
                # The canonical labels the surfaces resolved to. The query engine pairs
                # these with the two fields above to show "Resolved: 'He' -> Arun Mehta".
                canonical_subject=triple.subject,
                canonical_object=triple.object,
                triple_id=triple.triple_id,
                duplicate_count=triple.duplicate_count,
            )

        self._finalise_degrees(graph)
        return graph

    # -- internals ---------------------------------------------------------

    def _add_node(
        self,
        graph: nx.MultiDiGraph,
        entity_id: str,
        mentions: Mapping[str, tuple[int, list[str]]],
    ) -> None:
        # The registry holds two record shapes -- CanonicalEntity and LiteralNode -- and
        # LiteralNode has no `type` or `aliases` field. The registry accessors normalise
        # both, so nothing here reaches into record attributes directly.
        entity_type = self._registry.type_of(entity_id)
        count, documents = mentions.get(entity_id, (0, []))
        # No `id` attribute: the node key *is* the entity id. Storing it again would be
        # duplication that networkx's node_link_graph silently drops on load, since it
        # consumes `id` as the key -- leaving the round-trip lossy.
        graph.add_node(
            entity_id,
            label=self._registry.label_of(entity_id) or entity_id,
            type=entity_type.value if entity_type else "UNKNOWN",
            aliases=list(self._registry.aliases_of(entity_id)),
            mention_count=count,
            documents=documents,
        )

    @staticmethod
    def _mention_counts(
        resolutions: Iterable[MentionResolution],
    ) -> dict[str, tuple[int, list[str]]]:
        """Per entity: (resolved mention count, sorted list of documents it appeared in).

        Only resolved mentions count. An unresolved mention belongs to no node, so
        counting it would inflate whichever entity happens to share its surface form.
        """
        counts: dict[str, int] = defaultdict(int)
        documents: dict[str, set[str]] = defaultdict(set)
        for resolution in resolutions:
            if resolution.entity_id is None:
                continue
            counts[resolution.entity_id] += 1
            documents[resolution.entity_id].add(resolution.document_id)
        return {
            entity_id: (count, sorted(documents[entity_id]))
            for entity_id, count in counts.items()
        }

    @staticmethod
    def _finalise_degrees(graph: nx.MultiDiGraph) -> None:
        """Attach degrees after every edge exists.

        Computed in a second pass because a node's degree changes as later triples are
        added; reading ``graph.degree`` during construction would record stale values.
        """
        for entity_id, data in graph.nodes(data=True):
            data["in_degree"] = int(graph.in_degree(entity_id))
            data["out_degree"] = int(graph.out_degree(entity_id))
            data["degree"] = int(graph.degree(entity_id))
