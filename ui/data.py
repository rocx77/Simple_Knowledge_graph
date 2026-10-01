"""Cached data access for the Streamlit app.

The expensive objects are the spaCy pipeline and the FastCoref model, both of which are
cached as *resources* (shared, not copied). Everything they produce is serialisable, so
the resulting graph, triples and report are cached as *data*.

Rebuilding is deliberately a separate, explicit action: querying must never re-run NLP
(specification section 41).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.config import AppConfig
from src.entity_resolver import EntityRegistry
from src.pipeline import KnowledgeGraphPipeline

if TYPE_CHECKING:
    import networkx as nx

    from src.models import Triple


@dataclass(frozen=True)
class KnowledgeBase:
    """Everything the UI needs, loaded from the four artifacts.

    Deliberately a frozen dataclass of plain values so ``st.cache_data`` can hash it.
    """

    graph: nx.MultiDiGraph
    triples: tuple[Triple, ...]
    report: dict[str, Any]
    entities: tuple[dict[str, Any], ...]
    resolutions: tuple[dict[str, Any], ...]

    # -------------------------------------------------------------- KPI statistics
    @property
    def documents(self) -> int:
        return int(self.report.get("counts", {}).get("documents", 0))

    @property
    def mention_count(self) -> int:
        return int(self.report.get("counts", {}).get("mentions", 0))

    @property
    def unresolved_mentions(self) -> int:
        return int(self.report.get("counts", {}).get("unresolved_mentions", 0))

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    @property
    def relation_types(self) -> list[str]:
        return sorted({str(d.get("relation")) for _, _, d in self.graph.edges(data=True)})

    @property
    def entity_types(self) -> list[str]:
        return sorted({str(d.get("type")) for _, d in self.graph.nodes(data=True)})

    @property
    def canonical_entities(self) -> int:
        """Declared canonical entities present in the graph.

        The literal node is counted separately from the named entities because it is
        resolved by a relation rule and has no textual mention of its own.
        """
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") != "LITERAL")


def _read_triples(path: Path) -> tuple[Triple, ...]:
    """Rebuild Triple objects from the CSV artifact.

    The CSV is the contract with anyone consuming the artifacts, so the UI reads it back
    rather than reaching into the pipeline's in-memory state.
    """
    from src.models import Triple

    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    triples = []
    for row in rows:
        triples.append(
            Triple(
                triple_id=row["triple_id"],
                subject=row["subject"],
                subject_id=row["subject_id"],
                relation=row["relation"],
                object=row["object"],
                object_id=row["object_id"],
                document_id=row["document_id"],
                sentence_index=int(row["sentence_index"]),
                sentence=row["sentence"],
                original_subject=row["original_subject"],
                original_object=row["original_object"],
                coref_resolved=row.get("coref_resolved", "") == "True",
                confidence=float(row["confidence"]),
                rule_id=row["rule_id"],
                inferred=row.get("inferred", "") == "True",
            )
        )
    return tuple(triples)


@st.cache_resource(show_spinner="Loading models (spaCy + FastCoref)…")
def get_pipeline() -> KnowledgeGraphPipeline:
    """The pipeline, including the loaded spaCy and FastCoref models.

    ``cache_resource`` rather than ``cache_data``: the object holds large, unserialisable
    models that should be shared, not copied, between sessions.
    """
    return KnowledgeGraphPipeline(AppConfig.from_env())


@st.cache_data(show_spinner="Loading graph…", ttl=None, max_entries=2)
def load_knowledge_base(artifacts_dir: str | Path, rebuild_token: int = 0) -> KnowledgeBase:
    """Read the four artifacts into a :class:`KnowledgeBase`.

    ``rebuild_token`` exists only to invalidate this cache after a rebuild; bumping it is
    cheaper and clearer than calling ``clear()`` on this specific function.
    """
    del rebuild_token  # only ever read as a cache key
    from src.serialization import load_graph

    directory = Path(artifacts_dir)
    graph = load_graph(directory / "graph.json")
    triples = _read_triples(directory / "triples.csv")

    report_path = directory / "pipeline_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    entities_path = directory / "entities.json"
    entities: tuple[dict[str, Any], ...] = ()
    if entities_path.exists():
        payload = json.loads(entities_path.read_text(encoding="utf-8"))
        entities = tuple(payload.get("entities", payload if isinstance(payload, list) else []))

    resolutions = tuple(
        {
            "document_id": r.document_id,
            "mention": r.mention,
            "entity_id": r.entity_id,
            "entity": r.entity_label,
            "mention_source": getattr(r, "mention_source", ""),
            "source": r.source,
            "confidence": r.confidence,
            "is_anaphoric": getattr(r, "is_anaphoric", None),
        }
        for r in _last_resolutions()
    )
    return KnowledgeBase(
        graph=graph,
        triples=triples,
        report=report,
        entities=entities,
        resolutions=resolutions,
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def _last_resolutions() -> tuple[Any, ...]:
    """Resolutions from the most recent run, if this session performed one.

    The artifacts deliberately do not persist per-mention resolution detail, so the debug
    view shows it only after a rebuild in the same session. Reading the graph instead would
    mean re-inventing the pipeline's internals, which is exactly what the artifacts exist
    to prevent.
    """
    return tuple(st.session_state.get("last_resolutions", ()))


@st.cache_resource(show_spinner=False, max_entries=1)
def get_registry() -> EntityRegistry:
    return EntityRegistry.load()


@st.cache_data(show_spinner="Rebuilding knowledge graph…", ttl=None, max_entries=1)
def rebuild(token: int) -> dict[str, Any]:
    """Run the full pipeline and write the artifacts.

    ``token`` is the cache key: each press gets a new value, so this always re-runs.
    """
    del token
    pipeline = get_pipeline()
    messages: list[str] = []
    result = pipeline.run(progress_cb=messages.append)
    st.session_state["last_resolutions"] = tuple(result.resolutions)
    st.session_state["rebuild_token"] = st.session_state.get("rebuild_token", 0) + 1
    st.session_state["messages"] = messages
    return {
        "nodes": result.graph.number_of_nodes(),
        "edges": result.graph.number_of_edges(),
        "triples": len(result.triples),
        "messages": messages,
        "report": result.report.model_dump() if hasattr(result.report, "model_dump") else {},
    }


def artifacts_present(artifacts_dir: str | Path) -> bool:
    directory = Path(artifacts_dir)
    return (directory / "graph.json").exists() and (directory / "triples.csv").exists()
