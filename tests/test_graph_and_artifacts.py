"""S7/S8 acceptance: graph construction and the four artifacts.

Asserts on the artifacts on disk rather than on in-memory objects, because "the graph can
be rebuilt" and "the graph can be loaded" are two separate promises and only the second is
tested by reading the file back.
"""

from __future__ import annotations

import csv
import json

import networkx as nx
import pytest

from src.graph_builder import GraphBuilder
from src.serialization import TRIPLE_CSV_COLUMNS, load_graph, write_triples_csv

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.acceptance]


@pytest.fixture(scope="module")
def result(extraction_result, config, registry):
    from src.pipeline import KnowledgeGraphPipeline

    return KnowledgeGraphPipeline(config).run()


@pytest.fixture(scope="module")
def graph(result):
    return result.graph


class TestGraphConstruction:
    def test_node_and_edge_counts(self, graph):
        assert graph.number_of_nodes() == 86
        assert graph.number_of_edges() == 94

    def test_is_a_multidigraph(self, graph):
        """Two relations between one pair must stay separate edges."""
        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.is_directed()

    def test_every_node_has_the_required_attributes(self, graph):
        required = {
            "label", "type", "aliases", "mention_count", "documents",
            "in_degree", "out_degree", "degree",
        }
        for entity_id, data in graph.nodes(data=True):
            assert required <= set(data), f"{entity_id} missing {required - set(data)}"

    def test_every_edge_keeps_its_provenance(self, graph):
        required = {
            "relation", "confidence", "document_id", "sentence_id", "sentence",
            "inferred", "rule_id", "original_subject", "original_object", "triple_id",
        }
        for _u, _v, data in graph.edges(data=True):
            assert required <= set(data)
            assert data["document_id"], "provenance must never be collapsed away"
            assert 0.0 <= float(data["confidence"]) <= 1.0

    def test_degrees_are_consistent(self, graph):
        for entity_id, data in graph.nodes(data=True):
            assert data["in_degree"] == graph.in_degree(entity_id)
            assert data["out_degree"] == graph.out_degree(entity_id)
            assert data["degree"] == graph.degree(entity_id)
            assert data["degree"] == data["in_degree"] + data["out_degree"]

    def test_mention_counts_are_populated(self, graph):
        """Counts come from resolved mentions.

        LITERAL is the exception: the literal node exists because a rule resolved the
        phrase against the declared lexicon, not because mention detection produced a
        span for it, so it legitimately carries no mention of its own. It is in the graph
        purely as the object of `focuses_on`.
        """
        for entity_id, data in graph.nodes(data=True):
            if data["type"] == "LITERAL":
                assert data["mention_count"] == 0
                continue
            assert data["mention_count"] >= 1, f"{entity_id} has no resolved mentions"
            assert data["documents"], f"{entity_id} records no source document"

    def test_literal_node_comes_only_from_a_relation(self, graph):
        """The one literal is reachable only through `focuses_on`."""
        literals = [n for n, d in graph.nodes(data=True) if d["type"] == "LITERAL"]
        assert len(literals) == 17
        for literal in literals:
            assert graph.in_degree(literal) == 1
            assert {d["relation"] for *_, d in graph.in_edges(literal, data=True)} == {
            "focuses_on"
        }

    def test_no_symmetric_edges_are_stored(self, graph):
        """Inverse relations are a query concern; storing them would double the graph."""
        for subject, obj, key, data in graph.edges(keys=True, data=True):
            if data["relation"] not in {"collaborates_with", "partners_with"}:
                continue
            assert not graph.has_edge(obj, subject, key=key)


class TestArtifacts:
    def test_all_four_files_exist(self, result):
        for name in ("graph", "triples", "entities", "report"):
            assert getattr(result.artifacts, name).exists(), f"{name} was not written"

    def test_graph_json_round_trips(self, result):
        reloaded = load_graph(result.artifacts.graph)
        assert isinstance(reloaded, nx.MultiDiGraph)
        assert reloaded.number_of_nodes() == 86
        assert reloaded.number_of_edges() == 94
        assert {d["label"] for _, d in reloaded.nodes(data=True)} == {
            d["label"] for _, d in result.graph.nodes(data=True)
        }
        assert {d["triple_id"] for *_, d in reloaded.edges(keys=True, data=True)} == {
            d["triple_id"] for *_, d in result.graph.edges(keys=True, data=True)
        }
        # Node and edge attributes must survive serialisation, not just the shape.
        original = {d["label"]: d for _, d in result.graph.nodes(data=True)}
        for _node_id, data in reloaded.nodes(data=True):
            assert data == original[data["label"]]

    def test_graph_json_is_visjs_shaped(self, result):
        payload = json.loads(result.artifacts.graph.read_text(encoding="utf-8"))
        assert payload["directed"] is True
        assert payload["multigraph"] is True
        assert "nodes" in payload and "links" in payload, "vis.js reads 'links'"
        assert payload["graph"]["node_count"] == 86
        assert payload["graph"]["edge_count"] == 94

    def test_json_files_are_indent_two(self, result):
        for path in (result.artifacts.graph, result.artifacts.entities, result.artifacts.report):
            lines = path.read_text(encoding="utf-8").splitlines()
            assert lines[1].startswith('  "'), "specification requires indent=2"

    def test_triples_csv(self, result, tmp_path):
        with result.artifacts.triples.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 94
        assert tuple(rows[0]) == TRIPLE_CSV_COLUMNS
        assert {r["confidence"] for r in rows} <= {"1.00", "0.90", "0.80"}
        assert {r["confidence"] for r in rows} == {"1.00", "0.90", "0.80"}

    def test_entities_json_lists_every_declared_entity(self, result, registry):
        payload = json.loads(result.artifacts.entities.read_text(encoding="utf-8"))
        declared = len(registry.entities) + len(registry.literals)
        assert len(payload["entities"]) == declared
        assert {e["id"] for e in payload["entities"]} == {
            r.entity_id for r in list(registry.entities) + list(registry.literals)
        }

    def test_entities_json_flags_unconnected_entities(self, result):
        """An entity with no relations must be present and visibly inert, not dropped."""
        payload = json.loads(result.artifacts.entities.read_text(encoding="utf-8"))
        for entity in payload["entities"]:
            assert "has_edges" in entity
        inert = [e["id"] for e in payload["entities"] if not e["has_edges"]]
        assert inert == [], f"all declared entities are connected in this corpus: {inert}"

    def test_entities_json_records_unresolved_mentions(self, result):
        payload = json.loads(result.artifacts.entities.read_text(encoding="utf-8"))
        unresolved = payload["unresolved_mentions"]
        assert len(unresolved) == 2
        assert unresolved[0]["text"] == "GPU"

    def test_pipeline_report_contents(self, result):
        payload = json.loads(result.artifacts.report.read_text(encoding="utf-8"))
        assert payload["counts"]["nodes"] == 86
        assert payload["counts"]["triples"] == 94
        assert payload["counts"]["documents"] == 24
        assert payload["counts"]["mentions"] == 194
        assert payload["counts"]["unresolved_mentions"] == 2
        assert payload["relation_histogram"]["leads"] == 13
        assert payload["relation_histogram"]["contains"] == 2
        assert sum(payload["relation_histogram"].values()) == 94
        assert len(payload["coref_conflicts"]) == 1
        assert payload["versions"]["spacy"]
        assert payload["device"] in {"cpu", "cuda"}
        assert all(stage["ok"] for stage in payload["stages"])
        assert all(stage["elapsed_seconds"] >= 0 for stage in payload["stages"])

    def test_report_explains_the_skipped_candidate(self, result):
        payload = json.loads(result.artifacts.report.read_text(encoding="utf-8"))
        skipped = payload["skipped_candidates"]
        assert len(skipped) == 4
        assert skipped[0]["rule_id"] == "R_LEADS"
        assert "entity resolution work" in skipped[0]["reason"]


class TestBuilderInIsolation:
    def test_building_twice_is_idempotent(self, extraction_result, registry):
        """Same triples in, same graph out -- the builder must not accumulate state."""
        triples = extraction_result.triples
        graph_a = GraphBuilder(registry).build(triples)
        graph_b = GraphBuilder(registry).build(triples)
        assert graph_a.number_of_nodes() == graph_b.number_of_nodes() == 86
        assert graph_a.number_of_edges() == graph_b.number_of_edges() == 94

    def test_write_triples_csv_returns_row_count(self, extraction_result, tmp_path):
        path = tmp_path / "t.csv"
        assert write_triples_csv(path, extraction_result.triples) == 94


class TestNoStreamlitInSrc:
    def test_src_never_imports_streamlit(self):
        """S8 acceptance: the UI must not leak into the pipeline.

        Checked by parsing the AST rather than grepping, because `src/` legitimately
        mentions Streamlit in docstrings and in the logging formatter's noisy-library
        list. Only a real import would violate the layering rule.
        """
        import ast
        import pathlib

        offenders = []
        for path in sorted(pathlib.Path("src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "streamlit" or name.startswith("streamlit.") for name in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"src/ imports Streamlit: {offenders}"
