"""S7b -- write the four artifacts.

All JSON is written with ``indent=2`` because the specification requires the output to
stay human-readable; these files are meant to be opened and read, not just parsed.

``graph.json`` is node-link data, which is what PyVis and vis.js consume, and is also
directly loadable back into ``networkx.readwrite.json_graph.node_link_graph``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .config import AppConfig
from .entity_resolver import EntityRegistry
from .models import MentionResolution, Triple

#: Column order for ``triples.csv``. Fixed so downstream consumers can rely on it.
TRIPLE_CSV_COLUMNS = (
    "triple_id",
    "subject_id",
    "subject",
    "relation",
    "object_id",
    "object",
    "document_id",
    "sentence_index",
    "sentence",
    "original_subject",
    "original_object",
    "confidence",
    "coref_resolved",
    "inferred",
    "rule_id",
    "context_entity",
    "duplicate_count",
)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    graph: Path
    triples: Path
    entities: Path
    report: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "graph": str(self.graph),
            "triples": str(self.triples),
            "entities": str(self.entities),
            "report": str(self.report),
        }


@dataclass(slots=True)
class PipelineReport:
    """Accumulates everything ``pipeline_report.json`` needs while stages run.

    Stages append to :attr:`stages` and :attr:`warnings` as they execute, so the report is
    a record of what happened rather than a summary reconstructed afterwards.
    """

    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    device: str = "cpu"
    relation_histogram: dict[str, int] = field(default_factory=dict)
    unresolved_mentions: list[dict[str, Any]] = field(default_factory=list)
    skipped_candidates: list[dict[str, Any]] = field(default_factory=list)
    coref_conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "environment": self.environment,
            "versions": self.versions,
            "stages": self.stages,
            "counts": self.counts,
            "relation_histogram": self.relation_histogram,
            "warnings": self.warnings,
            "unresolved_mentions": self.unresolved_mentions,
            "coref_conflicts": self.coref_conflicts,
            "skipped_candidates": self.skipped_candidates,
        }


class ArtifactWriter:
    """Serialises pipeline output to disk."""

    def __init__(self, config: AppConfig, registry: EntityRegistry) -> None:
        self._config = config
        self._registry = registry

    def write_all(
        self,
        graph: nx.MultiDiGraph,
        triples: Sequence[Triple],
        resolutions: Sequence[MentionResolution],
        report: PipelineReport,
    ) -> ArtifactPaths:
        self._config.ensure_artifacts_dir()
        paths = ArtifactPaths(
            graph=self._config.graph_json,
            triples=self._config.triples_csv,
            entities=self._config.entities_json,
            report=self._config.pipeline_report_json,
        )
        write_json(paths.graph, graph_payload(graph))
        write_triples_csv(paths.triples, triples)
        write_json(paths.entities, _entities_payload(self._registry, graph, resolutions))
        write_json(paths.report, report.to_dict())
        return paths


# -- payload builders -------------------------------------------------------


def graph_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Node-link payload, JSON-safe.

    ``networkx.node_link_data`` leaves ``multigraph`` and ``directed`` flags out by
    default in some versions and emits sets for multigraph keys, so the shape is
    normalised explicitly rather than trusted.
    """
    data = nx.node_link_data(graph, edges="links")
    data["directed"] = True
    data["multigraph"] = True
    data.setdefault("graph", {})
    data["graph"].update(
        {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }
    )
    for link in data["links"]:
        link["confidence"] = float(link.get("confidence", 0.0))
    return data


def _entities_payload(
    registry: EntityRegistry,
    graph: nx.MultiDiGraph,
    resolutions: Sequence[MentionResolution],
) -> dict[str, Any]:
    """Canonical registry, mention inventory, and the alias index.

    Every declared entity appears, connected or not. ``has_edges`` makes an
    unconnected entity visibly inert rather than absent, which is the whole point of
    keeping the inventory honest.
    """
    mentions: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    for resolution in resolutions:
        record = {
            "document_id": resolution.document_id,
            "sentence_index": resolution.sentence_index,
            "text": resolution.mention,
            "start_char": resolution.start_char,
            "end_char": resolution.end_char,
            "source": resolution.source.value,
            "mention_source": resolution.mention_source.value,
            "resolved_to": resolution.entity_id,
            "resolved_label": resolution.resolved_label,
            "conflict": resolution.conflict,
        }
        if resolution.entity_id is None:
            unresolved.append(record)
        else:
            mentions.setdefault(resolution.entity_id, []).append(record)

    entities = []
    alias_index: dict[str, list[str]] = {}
    for record in list(registry.entities) + list(registry.literals):
        entity_id = record.entity_id
        # LiteralNode has no `type`/`canonical`/`aliases` field, so both record shapes go
        # through the registry accessors rather than attribute access.
        entity_type = registry.type_of(entity_id)
        entities.append(
            {
                "id": entity_id,
                "label": record.label,
                "type": entity_type.value if entity_type else "UNKNOWN",
                "aliases": list(registry.aliases_of(entity_id)),
                "description": record.description,
                "canonical": not registry.is_literal(entity_id),
                "has_edges": entity_id in graph,
                "mention_count": len(mentions.get(entity_id, [])),
                "documents": sorted(
                    {m["document_id"] for m in mentions.get(entity_id, [])}
                ),
            }
        )
        for alias in registry.aliases_of(entity_id):
            alias_index.setdefault(alias.casefold(), []).append(entity_id)

    return {
        "entities": entities,
        "alias_index": {k: sorted(v) for k, v in sorted(alias_index.items())},
        "mention_inventory": dict(sorted(mentions.items())),
        "unresolved_mentions": unresolved,
    }


# -- writers ----------------------------------------------------------------


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Read ``graph.json`` back into a ``MultiDiGraph``.

    ``edges="links"`` is mandatory here. The file uses ``links`` because that is what
    vis.js and PyVis expect, but networkx 3.7's ``node_link_graph`` defaults to reading
    ``edges`` and raises ``KeyError`` otherwise. Centralised so the UI and the tests
    cannot get it wrong independently.
    """
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return nx.node_link_graph(payload, edges="links")


def write_triples_csv(path: Path, triples: Iterable[Triple]) -> int:
    rows = list(triples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRIPLE_CSV_COLUMNS))
        writer.writeheader()
        for triple in rows:
            writer.writerow(
                {
                    "triple_id": triple.triple_id,
                    "subject_id": triple.subject_id,
                    "subject": triple.subject,
                    "relation": triple.relation.value,
                    "object_id": triple.object_id,
                    "object": triple.object,
                    "document_id": triple.document_id,
                    "sentence_index": triple.sentence_index,
                    "sentence": triple.sentence,
                    "original_subject": triple.original_subject,
                    "original_object": triple.original_object,
                    "confidence": f"{float(triple.confidence):.2f}",
                    "coref_resolved": str(triple.coref_resolved).lower(),
                    "inferred": str(triple.inferred).lower(),
                    "rule_id": triple.rule_id,
                    "context_entity": triple.context_entity,
                    "duplicate_count": triple.duplicate_count,
                }
            )
    return len(rows)


def relation_histogram(triples: Iterable[Triple]) -> dict[str, int]:
    counter: Counter[str] = Counter(t.relation.value for t in triples)
    return dict(sorted(counter.items()))
