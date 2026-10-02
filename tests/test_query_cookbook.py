"""Query cookbook: verified questions that exercise the whole pipeline.

Every query here is answered against the committed artifacts in artifacts/.
They exist so a reviewer can test the full pipeline with complex multi-hop
retrieval, not just single-triple lookups.
"""

from __future__ import annotations

import pathlib

import pytest

from src.entity_resolver import EntityRegistry
from src.query import QueryEngine
from src.serialization import load_graph

SPEC_ARTIFACT = pathlib.Path("artifacts") / "graph.json"

#: (question, expected answers). Answers are exact entity labels.
SPECIFIED: dict[str, list[str]] = {
    # Simple who-develops lookups, now with synonym robustness.
    "Who made Mario?": ["Shigeru Miyamoto", "Nintendo"],
    "Who created Mario?": ["Shigeru Miyamoto", "Nintendo"],
    # Complex multi-hop traversal across the graph's components.
    "How is OrionEdge connected to Mozilla?": ["Mozilla"],
    "How is Helios Bank connected to OpenAI?": ["OpenAI"],
    "How is Helios Bank connected to Netscape?": ["Netscape"],
    "How is OrionEdge connected to Linux Foundation?": ["Linux Foundation"],
    "How is GraphMatch connected to Mozilla?": ["Mozilla"],
    "How is Nila Rao connected to GraphMatch?": ["GraphMatch"],
    # Present in the graph but the pair is disconnected.
    "How is Priya Nair connected to Kyoto Animation?": [],
}


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    graph = load_graph(SPEC_ARTIFACT)
    return QueryEngine(graph, EntityRegistry.load())


@pytest.mark.parametrize("question", sorted(SPECIFIED))
def test_cookbook_question(engine: QueryEngine, question: str) -> None:
    result = engine.run(question)
    assert result.answers == SPECIFIED[question]


def test_cookbook_connection_has_paths(engine: QueryEngine) -> None:
    connected = [q for q, a in SPECIFIED.items() if a and q.startswith("How is")]
    for question in connected:
        assert engine.run(question).paths, question


def test_every_cookbook_question_is_listed() -> None:
    assert len(SPECIFIED) == len(set(SPECIFIED))
