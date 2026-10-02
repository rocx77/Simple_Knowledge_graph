"""Deterministic multi-hop query planner: AST, candidates, trace, failure steps."""

from __future__ import annotations

import pathlib

import pytest

from src.entity_resolver import EntityRegistry
from src.query import (
    EntityExpression,
    QueryEngine,
    QueryPlan,
    TriplePattern,
    VariableBinding,
)
from src.serialization import load_graph

SPEC_ARTIFACT = pathlib.Path("artifacts") / "graph.json"


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    if not SPEC_ARTIFACT.exists():
        pytest.skip("run `python -m src.pipeline` first")
    return QueryEngine(load_graph(SPEC_ARTIFACT), EntityRegistry.load())


def test_starship_founder_example(engine: QueryEngine) -> None:
    """'Who founded the company that developed Starship?' — the key example."""
    result = engine.run("Who founded the company that developed Starship?")
    assert result.answers == ["Elon Musk"]
    assert result.query_plan is not None
    relations = [p.relation.value for p in result.query_plan.patterns]
    assert relations == ["develops", "founded"]
    # Every matched step carries document provenance.
    assert result.execution_steps
    for step in result.execution_steps:
        if step.status == "matched":
            assert step.document_id
            assert step.sentence
            assert step.rule_id
    assert result.trace  # human-readable trace exists


def test_nasa_partner_products(engine: QueryEngine) -> None:
    result = engine.run("What does the organization that NASA partners with develop?")
    assert result.answers == ["Starship"]
    assert [p.relation.value for p in result.query_plan.patterns] == [
        "partners_with",
        "develops",
    ]


def test_type_constraint_blocks_wrong_type(engine: QueryEngine) -> None:
    """NASA uses Starship (a PRODUCT), not a company: the ?company variable
    with its ORGANIZATION constraint must fail at step 1, exactly."""
    result = engine.run("What does the company that NASA uses focus on?")
    assert result.answers == []
    assert "Step 1 failed" in result.message
    failed = [s for s in result.execution_steps if s.status == "failed"]
    assert len(failed) == 1
    assert failed[0].step == 1
    assert failed[0].pattern.relation.value == "uses"
    # The failure is surfaced in the trace, not just the message.
    assert any("Step 1 failed" in line for line in result.trace)


def test_failure_step_is_exact(engine: QueryEngine) -> None:
    result = engine.run("Who leads the organization that partnered with DeNA?")
    assert result.answers == []
    assert "Step 2 failed" in result.message
    failed = [s for s in result.execution_steps if s.status == "failed"]
    assert failed[0].step == 2
    assert failed[0].pattern.relation.value == "leads"


def test_ambiguous_phrase_produces_candidate_plans(engine: QueryEngine) -> None:
    """'works with' maps to collaborates_with AND partners_with in the synonym
    table; both candidate plans must be built and traced, not silently picked."""
    plan = engine.build_plan("What does the organization that works with NASA develop?")
    assert plan is not None
    assert plan.candidates, "expected ambiguity candidate plans"
    candidate_relations = {p.relation.value for p in plan.patterns}
    for candidate in plan.candidates:
        candidate_relations.update(p.relation.value for p in candidate.patterns)
    assert {"collaborates_with", "partners_with"} <= candidate_relations

    result = engine.run("What does the organization that works with NASA develop?")
    assert result.answers == ["Starship"]  # via the partners_with candidate
    assert "Ambiguous" in result.message
    # Both candidates appear in the trace.
    assert any("primary" in line for line in result.trace)
    assert any("partners_with" in line for line in result.trace)


def test_nested_relative_clause_recurses(engine: QueryEngine) -> None:
    """Two levels of 'that' nesting compiles to three conjunctive patterns."""
    result = engine.run(
        "Who founded the company that developed the product that NASA uses?"
    )
    assert result.answers == ["Elon Musk"]
    assert len(result.query_plan.patterns) == 3


def test_plan_ast_types(engine: QueryEngine) -> None:
    plan = engine.build_plan("Who founded the company that developed Starship?")
    assert isinstance(plan, QueryPlan)
    first = plan.patterns[0]
    assert isinstance(first, TriplePattern)
    assert isinstance(first.subject, EntityExpression)
    assert first.subject.is_variable
    assert first.object.entity_id == "product:starship"
    # VariableBinding immutability: bind returns a new object.
    b0 = VariableBinding()
    b1 = b0.bind("?company", "organization:spacex")
    assert b0.values == {}
    assert b1.get("?company") == "organization:spacex"


def test_parse_is_used_for_single_hop_and_semantic_is_separate(engine: QueryEngine) -> None:
    """Single-hop interpretation remains a QueryParse, not a plan."""
    result = engine.run("Who founded Aether Analytics?")
    assert result.parse is not None
    assert result.parse.intent.value == "relation_lookup"
    assert result.answers == ["Dr. Mira Sen"]
    # Multi-hop results carry a plan instead of a single-entity parse.
    multi = engine.run("Who founded the company that developed Starship?")
    assert multi.parse is None
    assert multi.query_plan is not None
    # Connection queries are untouched.
    conn = engine.run("How is Arun Mehta connected to OrionEdge?")
    assert conn.paths
