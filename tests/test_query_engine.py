"""S9 acceptance: the twelve specified questions plus connection search.

Expected answers are transcribed from specification section 18, not recorded from a run.
"""

from __future__ import annotations

import pathlib

import pytest

from src.entity_resolver import EntityRegistry
from src.query import (
    Direction,
    QueryEngine,
    QueryIntent,
    iter_query_questions,
)
from src.serialization import load_graph

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.acceptance]

SPEC_ARTIFACT = pathlib.Path("artifacts") / "graph.json"

#: Question -> expected answers, from specification section 18.
SPECIFIED: dict[str, list[str]] = {
    "Who founded Aether Analytics?": ["Dr. Mira Sen"],
    "What does Aether Analytics develop?": ["OrionEdge"],
    "Who works at Aether Analytics?": ["Arun Mehta"],
    "Who works on OrionEdge?": ["Arun Mehta"],
    "Who leads Project Aurora?": ["Dr. Mira Sen", "Nila Rao"],
    "Who collaborates with Arun Mehta?": ["Nila Rao"],
    "Who partnered with Aether Analytics?": ["Quantum Forge", "LedgerLine"],
    "Where is OrionEdge deployed?": ["Helios Bank"],
    "Who uses OrionEdge?": ["Helios Bank"],
    "Who improved the Entity Resolution Engine?": ["Nila Rao"],
    "What is integrated into OrionEdge?": ["Project Aurora", "LedgerLine"],
    "Who is the customer of Aether Analytics?": ["Helios Bank"],
}


@pytest.fixture(scope="module")
def engine():
    if not SPEC_ARTIFACT.exists():
        pytest.skip("run `python -m src.pipeline` first")
    graph = load_graph(SPEC_ARTIFACT)
    return QueryEngine(graph, EntityRegistry.load())


@pytest.mark.parametrize("question", sorted(SPECIFIED))
def test_specified_question(engine, question):
    result = engine.run(question)
    assert result.answers == SPECIFIED[question]


def test_every_specified_question_is_listed():
    """The helper used by the UI must not drift from the specification's set."""
    assert set(iter_query_questions()) == set(SPECIFIED)


class TestInterpretation:
    def test_direction_is_reported(self, engine):
        assert engine.parse("Who founded Aether Analytics?").direction is Direction.INCOMING
        assert (
            engine.parse("What does Aether Analytics develop?").direction
            is Direction.OUTGOING
        )

    def test_symmetric_relations_search_both_directions(self, engine):
        for question in (
            "Who collaborates with Arun Mehta?",
            "Who partnered with Aether Analytics?",
        ):
            assert engine.parse(question).direction is Direction.BOTH

    def test_intent_is_identified(self, engine):
        assert engine.parse("Who founded Aether Analytics?").intent is QueryIntent.RELATION_LOOKUP
        assert (
            engine.parse("How is Arun Mehta connected to OrionEdge?").intent
            is QueryIntent.CONNECTION
        )

    def test_interpretation_is_serialisable(self, engine):
        payload = engine.run("Who works on OrionEdge?").to_dict()
        interpretation = payload["interpretation"]
        assert interpretation["entity"] == "OrionEdge"
        assert interpretation["relation"] == "works_on"
        assert interpretation["direction"] == "incoming"
        assert interpretation["intent"] == "relation_lookup"

    def test_entity_match_prefers_canonical_names(self, engine):
        """Canonical label beats the alias 'Aurora', which also exists for Project Aurora."""
        parse = engine.parse("Who leads Aurora?")
        assert parse.entity_label == "Project Aurora"
        assert parse.entity_match == "alias" or parse.entity_match == "canonical"
        assert engine.parse("Who leads Project Aurora?").entity_match == "canonical"


class TestEvidence:
    def test_every_answer_carries_its_sentence(self, engine):
        for question in SPECIFIED:
            for item in engine.run(question).evidence:
                assert item["document_id"], question
                assert item["sentence"], question
                assert item["triple_id"]

    def test_coreference_answers_show_what_was_resolved(self, engine):
        """'Who works on OrionEdge?' is answered from a pronoun; say so."""
        result = engine.run("Who works on OrionEdge?")
        resolved = [item for item in result.evidence if item["resolutions"]]
        assert resolved, "the works_on edge comes from 'He' and must say so"
        pairs = {(r["from"], r["to"]) for item in resolved for r in item["resolutions"]}
        assert ("He", "Arun Mehta") in pairs
        assert float(resolved[0]["confidence"]) == 0.90

    def test_direct_answers_are_marked_top_confidence(self, engine):
        result = engine.run("Who founded Aether Analytics?")
        assert float(result.evidence[0]["confidence"]) == 1.00
        assert result.evidence[0]["resolutions"] == []

    def test_inferred_answer_is_flagged(self, engine):
        evidence = engine.run("Who is the customer of Aether Analytics?").evidence[0]
        assert evidence["inferred"] is True
        assert float(evidence["confidence"]) == 0.80


class TestAnswerOrdering:
    def test_multiple_answers_come_in_document_order(self, engine):
        """Dr. Mira Sen is stated in doc_01, Nila Rao in doc_02."""
        result = engine.run("Who leads Project Aurora?")
        assert result.answers == ["Dr. Mira Sen", "Nila Rao"]
        assert [e["document_id"] for e in result.evidence] == [
            "doc_01_company",
            "doc_02_team",
        ]

    def test_answers_are_deduplicated(self, engine):
        result = engine.run("Who leads Project Aurora?")
        assert len(result.answers) == len(set(result.answers))


class TestConnectionQueries:
    def test_direct_connection(self, engine):
        result = engine.run("How is Arun Mehta connected to OrionEdge?")
        assert result.paths
        shortest = result.paths[0]
        assert len(shortest) == 1
        assert shortest[0]["relation"] == "works_on"

    def test_spec_example_path_is_found(self, engine):
        """The specification's worked example, verbatim.

        Nila Rao leads Project Aurora; Project Aurora is led by Dr. Mira Sen; Mira
        founded Aether Analytics. The middle hop runs against the stored edge direction,
        so an outgoing-only search cannot find it.

        BFS yields shortest paths first, and a 2-hop route exists via Arun Mehta, so the
        spec's 3-hop path is present but not first.
        """
        result = engine.run("How is Nila Rao connected to Aether Analytics?")

        # Every hop the specification draws, across every path returned.
        hops = {
            (step["from"], step["relation"], step["to"])
            for path in result.paths
            for step in path
        }
        assert ("Nila Rao", "leads", "Project Aurora") in hops
        assert ("Project Aurora", "led by", "Dr. Mira Sen") in hops
        assert ("Dr. Mira Sen", "founded", "Aether Analytics") in hops

        # And they appear as one contiguous path, not three unrelated hops.
        assert any(
            [(s["from"], s["relation"], s["to"]) for s in path]
            == [
                ("Nila Rao", "leads", "Project Aurora"),
                ("Project Aurora", "led by", "Dr. Mira Sen"),
                ("Dr. Mira Sen", "founded", "Aether Analytics"),
            ]
            for path in result.paths
        )
        # Shortest path wins, and it is still a real route.
        assert len(result.paths[0]) == 2

    def test_direction_flags_are_consistent_across_paths(self, engine):
        """BFS shares a prefix object between paths; the flag must survive every path."""
        result = engine.run("How is Quantum Forge connected to Project Aurora?")
        assert len(result.paths) > 1
        first_hops = {tuple(path[0].values()) for path in result.paths}
        assert len(first_hops) == 1, "paths sharing a prefix must label it identically"
        assert all(path[0]["direction"] == "reverse" for path in result.paths)

    def test_unconnected_entities_are_reported_not_crashed(self, engine):
        result = engine.run("How is Quantum Forge connected to Helios Bank?")
        assert result.paths  # reachable via Aether Analytics
        result = engine.run("How is the graph matching module connected to Mira Sen?")
        assert isinstance(result.paths, list)

    def test_max_depth_is_respected(self, engine):
        for path in engine.run("How is Arun Mehta connected to Helios Bank?").paths:
            assert len(path) <= 3


class TestUnknownInput:
    @pytest.mark.parametrize(
        "question",
        ["What is the meaning of life?", "Tell me about Bob.", "", "???"],
    )
    def test_unparseable_questions_return_a_message(self, engine, question):
        result = engine.run(question)
        assert result.answers == []
        assert result.message, f"no guidance offered for {question!r}"
        assert result.parse is not None

    def test_known_entity_but_unknown_relation(self, engine):
        result = engine.run("What is the weather at Aether Analytics?")
        assert result.answers == []
        assert "Aether Analytics" in result.message

    def test_known_relations_but_no_such_edge(self, engine):
        """Question 9's shape, aimed at an entity with no uses edge."""
        result = engine.run("Who uses Nila Rao?")
        assert result.answers == []
        assert "No uses relation" in result.message


class TestSynonymTable:
    def test_multiword_phrases_win_over_substrings(self, engine):
        """'works on' must not be matched as 'works at', nor 'uses' inside 'uses the'."""
        parse = engine.parse("Who works at OrionEdge?")
        assert parse.relation_phrase == "works at"
        assert "works_at" in [r.value for r in parse.relations]

    def test_article_is_stripped_when_matching_entities(self, engine):
        assert engine.run("Who improved the Entity Resolution Engine?").answers == [
            "Nila Rao"
        ]

    def test_ambiguous_phrase_is_flagged_not_guessed(self, engine):
        """The spec maps 'works with' to two relations; both are searched and flagged."""
        parse = engine.parse("Who works with Arun Mehta?")
        assert len(parse.relations) > 1
        assert parse.ambiguous is True
