"""S9 -- natural-language query parsing and execution over the graph.

Deterministic throughout: no embeddings, no fuzzy matching, no model inference. A query
is resolved by finding a declared entity surface and a relation phrase from a fixed
synonym table, then running one directed or undirected edge search.

Direction is decided by *where the entity sits relative to the relation phrase* rather
than by interrogative words. That single rule covers all twelve specified questions:

    "Who founded Aether Analytics?"    entity after  the phrase  ->  incoming
    "What does Aether Analytics develop?"  entity before the phrase  ->  outgoing

It is more robust than keying off "who"/"what", because the two coincide only by
accident: "Where is OrionEdge deployed?" asks about a thing but still needs an outgoing
search from OrionEdge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx

from .entity_resolver import EntityRegistry
from .models import RelationName

#: Relation phrase -> canonical relation, from specification section 19.
#:
#: A strict superset of the specification's table: "deployed" and "is deployed" are added
#: because query 8 ("Where is OrionEdge deployed?") contains none of the listed forms.
RELATION_SYNONYMS: dict[RelationName, tuple[str, ...]] = {
    RelationName.FOUNDED: ("founded", "founder of", "started", "created", "co-founded"),
    RelationName.DEVELOPS: ("develops", "developed", "develop", "builds", "built", "makes"),
    RelationName.WORKS_ON: ("works on", "working on", "contributes to", "worked on"),
    RelationName.WORKS_AT: ("works at", "works for", "working at", "joined", "employee of"),
    RelationName.LEADS: ("leads", "leading", "leader of", "headed by", "heads", "lead of"),
    RelationName.COLLABORATES_WITH: (
        "collaborates with", "collaborate with", "collaborated with", "collaborates",
        # "works with" is in the specification's table under both collaborates_with and
        # partners_with. Kept here in both, and surfaced via AMBIGUOUS_PHRASES rather than
        # silently resolved to one of them.
        "works with",
    ),
    RelationName.PARTNERS_WITH: (
        "partnered with", "partners with", "partner with", "partnered", "partners",
        "works with",
    ),
    RelationName.DEPLOYED_AT: (
        "deployed at", "deployed in", "runs at", "is deployed", "was deployed",
        "deployed", "runs in",
    ),
    RelationName.USES: ("uses", "using", "use of", "utilizes", "utilises", "use"),
    RelationName.IMPROVES: ("improves", "improved", "improving", "enhanced", "improves on"),
    RelationName.INTEGRATED_INTO: (
        "integrated into", "integrated in", "integrates into", "integrated with",
        "integrated",
    ),
    RelationName.CUSTOMER_OF: ("customer of", "client of", "customers of"),
    RelationName.FOCUSES_ON: ("focuses on", "focus on", "focused on", "focuses"),
    RelationName.CONTAINS: ("contains", "contain", "includes", "comprises", "consists of"),
}

#: Stored once, in the stated direction; searched both ways.
SYMMETRIC_RELATIONS = frozenset({RelationName.COLLABORATES_WITH, RelationName.PARTNERS_WITH})

#: Matched phrases that legitimately belong to two relations, per the specification's own
#: table ("works with" is listed under both collaborates_with and partners_with, and
#: "develops" under both develops and works_on). Surfaced in the result rather than
#: silently picking one.
AMBIGUOUS_PHRASES = frozenset({"works with", "works on", "develops", "developed"})


class QueryIntent(str, Enum):
    RELATION_LOOKUP = "relation_lookup"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class Direction(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class QueryParse:
    """The interpretation shown in the UI's INTERPRETATION panel."""

    raw: str
    intent: QueryIntent
    entity_id: str | None = None
    entity_label: str | None = None
    entity_surface: str | None = None
    entity_match: str = "none"
    relations: tuple[RelationName, ...] = ()
    relation_phrase: str = ""
    direction: Direction = Direction.OUTGOING
    other_entity_id: str | None = None
    other_entity_label: str | None = None
    max_depth: int = 3
    ambiguous: bool = False

    @property
    def understood(self) -> bool:
        if self.intent is QueryIntent.CONNECTION:
            # A connection query names a relation phrase as a connector ("connected to"),
            # so it legitimately has no `relations` at all.
            return self.entity_id is not None and self.other_entity_id is not None
        return self.intent is not QueryIntent.UNKNOWN and bool(self.relations)


@dataclass(slots=True)
class QueryResult:
    """What specification section 21 asks the engine to return."""

    answers: list[str] = field(default_factory=list)
    matched_triples: list[tuple[str, str, str]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    paths: list[list[dict[str, str]]] = field(default_factory=list)
    parse: QueryParse | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answers": self.answers,
            "matched_triples": self.matched_triples,
            "evidence": self.evidence,
            "paths": self.paths,
            "interpretation": _interpretation(self.parse) if self.parse else {},
            "message": self.message,
        }


class QueryEngine:
    """Parses a question and answers it from a loaded graph."""

    def __init__(self, graph: nx.MultiDiGraph, registry: EntityRegistry) -> None:
        self._graph = graph
        self._registry = registry
        # Longest surface first so "Entity Resolution Engine" wins over a bare "Engine"
        # and multiword aliases are never shadowed by their own tail.
        self._surfaces = self._build_surface_index()

    # -- public API --------------------------------------------------------

    def run(self, question: str) -> QueryResult:
        parse = self.parse(question)
        if not parse.understood:
            return QueryResult(
                parse=parse,
                message=_no_match_message(parse),
            )
        if parse.intent is QueryIntent.CONNECTION:
            return self._connection(parse)
        return self._relation_lookup(parse)

    def parse(self, question: str) -> QueryParse:
        text = question.strip()
        lowered = _normalise(text)
        match, label, surface, how = self._match_entity(lowered)

        if _CONNECTION_RE.search(lowered) and match is not None:
            other, other_label, other_surface, _ = self._match_entity(
                lowered.replace(surface or "", " ", 1)
            )
            if other is not None and other != match:
                # Whichever entity is named first in the question is the path's source, so
                # the answer reads in the order the user wrote it. Entity surfaces are
                # matched longest-first, so "Aether Analytics" wins over "Nila Rao"
                # regardless of which appears first in the sentence.
                source, source_label = match, label
                sink, sink_label = other, other_label
                if other_surface and surface and lowered.find(other_surface) < lowered.find(surface):
                    source, source_label = other, other_label
                    sink, sink_label = match, label
                return QueryParse(
                    raw=text,
                    intent=QueryIntent.CONNECTION,
                    entity_id=source,
                    entity_label=source_label,
                    entity_surface=None,
                    entity_match=how,
                    other_entity_id=sink,
                    other_entity_label=sink_label,
                )

        relations, phrase = self._match_relation(lowered)
        if match is None or not relations:
            return QueryParse(
                raw=text,
                intent=QueryIntent.UNKNOWN,
                entity_id=match,
                entity_label=label,
                entity_surface=surface,
                entity_match=how,
                relations=relations,
                relation_phrase=phrase,
            )

        direction = self._direction(label, surface, phrase, relations, lowered)
        return QueryParse(
            raw=text,
            intent=QueryIntent.RELATION_LOOKUP,
            entity_id=match,
            entity_label=label,
            entity_surface=surface,
            entity_match=how,
            relations=relations,
            relation_phrase=phrase,
            direction=direction,
            ambiguous=len(relations) > 1 or phrase in AMBIGUOUS_PHRASES,
        )

    # -- relation lookup ---------------------------------------------------

    def _relation_lookup(self, parse: QueryParse) -> QueryResult:
        assert parse.entity_id is not None
        results: list[tuple[str, RelationName, dict[str, Any]]] = []
        for relation in parse.relations:
            results.extend(self._search(parse.entity_id, relation, parse.direction))

        # Two orderings matter here, for different reasons.
        #
        # Deduplicate: one answer per entity, so a relation asserted by several documents
        # is not listed twice.
        #
        # Then sort by where the evidence appears. `in_edges` yields predecessors in
        # adjacency-insertion order, which has nothing to do with the documents; sorting
        # by (document_id, sentence_id) makes "Who leads Project Aurora?" read in
        # document order -- Dr. Mira Sen from doc_01 before Nila Rao from doc_02 -- which
        # is what a reader expects and what the specification lists.
        best: dict[str, tuple[str, RelationName, dict[str, Any]]] = {}
        for answer, relation, data in results:
            current = best.get(answer)
            if current is None or _evidence_order(data) < _evidence_order(current[2]):
                best[answer] = (answer, relation, data)

        ordered = sorted(best.values(), key=lambda item: _evidence_order(item[2]))
        answers = [answer for answer, _relation, _data in ordered]
        triples = [
            (self._label(parse.entity_id), relation.value, answer)
            for answer, relation, _ in ordered
        ]
        evidence = [self._evidence(answer, data) for answer, _r, data in ordered]
        if not answers:
            return QueryResult(parse=parse, message=_empty_message(parse))
        return QueryResult(answers=answers, matched_triples=triples, evidence=evidence, parse=parse)

    def _search(
        self, entity_id: str, relation: RelationName, direction: Direction
    ) -> list[tuple[str, RelationName, dict[str, Any]]]:
        """Edges of ``relation`` touching ``entity_id``; the answer is the *other* endpoint.

        ``graph.in_edges(n)`` yields ``(u, n)``, so the answer is ``u``; ``out_edges(n)``
        yields ``(n, v)``, so the answer is ``v``. Getting these two the wrong way round
        silently returns the queried entity itself for every question.
        """
        found: list[tuple[str, RelationName, dict[str, Any]]] = []
        if direction in (Direction.INCOMING, Direction.BOTH):
            for u, _v, data in self._graph.in_edges(entity_id, data=True):
                if data.get("relation") == relation.value:
                    found.append((self._label(u), relation, data))
        if direction in (Direction.OUTGOING, Direction.BOTH):
            for _u, v, data in self._graph.out_edges(entity_id, data=True):
                if data.get("relation") == relation.value:
                    found.append((self._label(v), relation, data))
        return found

    # -- connection --------------------------------------------------------

    def _connection(self, parse: QueryParse) -> QueryResult:
        assert parse.entity_id is not None and parse.other_entity_id is not None
        paths: list[list[dict[str, str]]] = []
        for path_nodes, path_edges in self._bfs(
            parse.entity_id, parse.other_entity_id, parse.max_depth
        ):
            steps: list[dict[str, str]] = []
            # `get`, never `pop`: BFS reuses one prefix step object across every path that
            # descends through it, so consuming the flag here silently mislabels the
            # direction of all later paths sharing that prefix.
            for index, (source, target) in enumerate(
                zip(path_nodes, path_nodes[1:], strict=False)
            ):
                edge = path_edges[index]
                forward = bool(edge.get("_forward", True))
                relation = str(edge.get("relation", ""))
                steps.append(
                    {
                        "from": self._label(source),
                        "relation": relation if forward else _inverse_phrase(relation),
                        "to": self._label(target),
                        "document_id": str(edge.get("document_id", "")),
                        "direction": "forward" if forward else "reverse",
                    }
                )
            paths.append(steps)

        if not paths:
            return QueryResult(
                parse=parse,
                message=(
                    f"No path of length {parse.max_depth} or less connects "
                    f"{parse.entity_label} to {parse.other_entity_label}."
                ),
            )
        return QueryResult(
            answers=[parse.other_entity_label or ""],
            paths=paths,
            parse=parse,
            message=f"Found {len(paths)} path(s) of length {parse.max_depth} or less.",
        )

    def _bfs(
        self, source: str, target: str, max_depth: int
    ) -> Iterator[tuple[list[str], list[dict[str, Any]]]]:
        """Every path up to ``max_depth`` hops, traversing edges in **both** directions.

        Traversal is undirected because the specification's own example requires it: to
        connect Nila Rao to Aether Analytics it goes

            Nila Rao --leads--> Project Aurora <--leads-- Dr. Mira Sen
                        --founded--> Aether Analytics

        which hops backwards along the second `leads` edge. Each step records whether it
        ran with or against the stored direction, so the UI can label the hop "led by"
        rather than drawing an arrow the data does not support.

        Plain BFS with a visited set, per the specification's instruction not to build
        multi-hop reasoning. The visited set keeps symmetric relations from looping.
        """
        if source == target:
            return
        seen: set[str] = {source}
        frontier: list[tuple[str, list[str], list[dict[str, Any]]]] = [(source, [source], [])]
        for _depth in range(max_depth):
            nxt: list[tuple[str, list[str], list[dict[str, Any]]]] = []
            for node, node_path, edge_path in frontier:
                for _u, v, key, data in self._graph.out_edges(node, keys=True, data=True):
                    step = {**data, "key": key, "_forward": True}
                    if v == target:
                        yield node_path + [v], edge_path + [step]
                    elif v not in seen:
                        seen.add(v)
                        nxt.append((v, node_path + [v], edge_path + [step]))
                for u, _v, key, data in self._graph.in_edges(node, keys=True, data=True):
                    if u in seen:
                        continue
                    step = {**data, "key": key, "_forward": False}
                    if u == target:
                        yield node_path + [u], edge_path + [step]
                    else:
                        seen.add(u)
                        nxt.append((u, node_path + [u], edge_path + [step]))
            frontier = nxt

    # -- matching ----------------------------------------------------------

    def _build_surface_index(self) -> list[tuple[str, str, str, str]]:
        """(normalised surface, entity_id, label, match kind), longest surface first.

        Match kinds encode the precedence in specification section 20: canonical names
        first, then aliases, then lowercased forms.
        """
        entries: list[tuple[str, str, str, str]] = []
        records = list(self._registry.entities) + list(self._registry.literals)
        for record in records:
            canonical = _normalise(record.label)
            entries.append((canonical, record.entity_id, record.label, "canonical"))
            for alias in self._registry.aliases_of(record.entity_id):
                entries.append((_normalise(alias), record.entity_id, record.label, "alias"))
        # Deduplicate identical (surface, id) keeping the strongest kind.
        best: dict[tuple[str, str], tuple[str, str, str, str]] = {}
        for surface, entity_id, label, kind in entries:
            if not surface:
                continue
            key = (surface, entity_id)
            current = best.get(key)
            if current is None or _KIND_RANK[kind] < _KIND_RANK[current[3]]:
                best[key] = (surface, entity_id, label, kind)
        return sorted(best.values(), key=lambda e: (-len(e[0]), _KIND_RANK[e[3]]))

    def _match_entity(self, lowered: str) -> tuple[str | None, str | None, str | None, str]:
        for surface, entity_id, label, kind in self._surfaces:
            if re.search(rf"(?<![\w]){re.escape(surface)}(?![\w])", lowered):
                return entity_id, label, surface, kind
        return None, None, None, "none"

    @staticmethod
    def _match_relation(lowered: str) -> tuple[tuple[RelationName, ...], str]:
        """All relations whose phrase appears, with the longest phrase that matched.

        Every match is collected rather than stopping at the first, because the
        specification's own synonym table maps one phrase to two relations.
        """
        hits: list[tuple[int, str, RelationName]] = []
        for relation, phrases in RELATION_SYNONYMS.items():
            for phrase in phrases:
                needle = _normalise(phrase)
                position = lowered.find(needle)
                if position >= 0:
                    hits.append((position, needle, relation))
        if not hits:
            return (), ""
        # Prefer the earliest match; among ties prefer the longest phrase.
        hits.sort(key=lambda hit: (hit[0], -len(hit[1])))
        first_position = hits[0][0]
        at_position = [hit for hit in hits if hit[0] == first_position]
        phrase = max(at_position, key=lambda hit: len(hit[1]))[1]
        relations: list[RelationName] = []
        for hit in at_position:
            if hit[2] not in relations:
                relations.append(hit[2])
        return tuple(relations), phrase

    def _direction(
        self,
        label: str | None,
        surface: str | None,
        phrase: str,
        relations: tuple[RelationName, ...],
        lowered: str,
    ) -> Direction:
        if any(relation in SYMMETRIC_RELATIONS for relation in relations):
            return Direction.BOTH
        needle = surface or (label or "").casefold()
        entity_at = lowered.find(needle) if needle else -1
        phrase_at = lowered.find(phrase)
        if entity_at >= 0 and phrase_at >= 0:
            # Entity named before the predicate is the subject ("What does Aether
            # Analytics develop?"), after it is the object ("Who founded Aether
            # Analytics?").
            return Direction.OUTGOING if entity_at < phrase_at else Direction.INCOMING
        # If the phrase cannot be located after normalisation, treat the entity as the
        # subject. Wrong answers here are visible in the UI's INTERPRETATION panel.
        return Direction.OUTGOING

    # -- helpers -----------------------------------------------------------

    def _label(self, entity_id: str) -> str:
        if entity_id in self._graph:
            return self._graph.nodes[entity_id].get("label", entity_id)
        return self._registry.label_of(entity_id) or entity_id

    @staticmethod
    def _evidence(answer: str, data: dict[str, Any]) -> dict[str, Any]:
        """Evidence block for specification section 23.

        ``resolutions`` lists only the endpoints where the document used a *reference*
        rather than a name. That is the part that demonstrates the value of coreference,
        so the UI can print "Resolved: 'He' -> Arun Mehta".
        """
        subject = str(data.get("original_subject", ""))
        obj = str(data.get("original_object", ""))
        resolutions: list[dict[str, str]] = []
        if bool(data.get("coref_resolved", False)):
            if subject and subject != data.get("canonical_subject", subject):
                resolutions.append({"from": subject, "to": str(data.get("canonical_subject", subject))})
            if obj and obj != data.get("canonical_object", obj):
                resolutions.append({"from": obj, "to": str(data.get("canonical_object", obj))})
        return {
            "document_id": data.get("document_id", ""),
            "sentence": data.get("sentence", ""),
            "sentence_id": data.get("sentence_id", 0),
            "rule_id": data.get("rule_id", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "coref_resolved": bool(data.get("coref_resolved", False)),
            "inferred": bool(data.get("inferred", False)),
            "triple_id": data.get("triple_id", ""),
            "original_subject": subject,
            "original_object": obj,
            "resolutions": resolutions,
            "answer": answer,
        }


def _evidence_order(data: dict[str, Any]) -> tuple[str, int, str]:
    """Sort key placing evidence in document order."""
    return (str(data.get("document_id", "")), int(data.get("sentence_id", 0)), str(data.get("triple_id", "")))


#: Readable label for traversing an edge backwards, so a reverse hop reads as English
#: rather than as an arrow the data contradicts. The specification's own example for
#: "How is Nila Rao connected to Aether Analytics?" draws exactly this: an upward hop
#: labelled "led by".
INVERSE_PHRASES: dict[str, str] = {
    "leads": "led by",
    "founded": "founded by",
    "develops": "developed by",
    "works_at": "employs",
    "works_on": "worked on by",
    "contains": "contained in",
    "focuses_on": "focus of",
    "improves": "improved by",
    "integrated_into": "integrates",
    "deployed_at": "deployed",
    "uses": "used by",
    "customer_of": "has customer",
    "collaborates_with": "collaborates with",
    "partners_with": "partners with",
}


def _inverse_phrase(relation: str) -> str:
    return INVERSE_PHRASES.get(relation, f"reverse of {relation}")


_CONNECTION_RE = re.compile(r"\b(how|connected|connection|connect|linked|path)\b")

_KIND_RANK = {"canonical": 0, "alias": 1, "lowercase": 2}


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace.

    A leading article is stripped from surfaces so "the Entity Resolution Engine" matches
    the canonical name "Entity Resolution Engine".
    """
    lowered = text.casefold()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"^the ", "", lowered)


def _interpretation(parse: QueryParse) -> dict[str, Any]:
    return {
        "entity": parse.entity_label,
        "entity_match": parse.entity_match,
        "relation": ", ".join(r.value for r in parse.relations),
        "relation_phrase": parse.relation_phrase,
        "direction": parse.direction.value,
        "intent": parse.intent.value,
        "ambiguous": parse.ambiguous,
    }


def _no_match_message(parse: QueryParse) -> str:
    if parse.entity_id is None:
        return (
            "I could not find an entity in that question. Try a canonical name such as "
            '"Aether Analytics", "OrionEdge" or "Project Aurora".'
        )
    if not parse.relations:
        return (
            f'I recognised "{parse.entity_label}" but not the relation. Try phrasing like '
            '"Who founded Aether Analytics?" or "What does Aether Analytics develop?".'
        )
    return "I could not interpret that question."


def _empty_message(parse: QueryParse) -> str:
    relation = ", ".join(r.value for r in parse.relations)
    return f"No {relation} relation found for {parse.entity_label}."


def iter_query_questions() -> Iterable[str]:
    """The twelve specified questions, used by the UI's example list and by tests."""
    return (
        "Who founded Aether Analytics?",
        "What does Aether Analytics develop?",
        "Who works at Aether Analytics?",
        "Who works on OrionEdge?",
        "Who leads Project Aurora?",
        "Who collaborates with Arun Mehta?",
        "Who partnered with Aether Analytics?",
        "Where is OrionEdge deployed?",
        "Who uses OrionEdge?",
        "Who improved the Entity Resolution Engine?",
        "What is integrated into OrionEdge?",
        "Who is the customer of Aether Analytics?",
    )
