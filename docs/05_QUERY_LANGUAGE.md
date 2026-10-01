# 05 — Query language

Authoritative description of the deterministic query parser and executor.
Implementation: `src/query.py`. Acceptance tests: `tests/test_query_engine.py`.

This is **not** natural-language question answering. It is a small deterministic parser
that recognises a closed set of templates and executes them directly on the NetworkX graph
(spec §17). There is no free-text generation, no ranking, and no fallback that guesses.

---

## 1. Parse result

```python
@dataclass(frozen=True)
class QueryParse:
    raw: str
    intent: QueryIntent              # relation_lookup | connection | unknown
    entity_id: str | None
    entity_label: str | None
    entity_surface: str | None       # what the user typed
    entity_match: str | None         # canonical | alias | anaphor | none
    relations: tuple[RelationName, ...]   # may hold more than one — see §4
    relation_phrase: str | None      # as typed, for display
    direction: Direction             # outgoing | incoming | both
    other_entity_id: str | None      # connection queries only
    other_entity_label: str | None
    max_depth: int                   # connection queries only
    ambiguous: bool                  # see §4
```

The UI shows every field of this object. Showing the parse is the point: a reviewer should
be able to see that the system understood the question, and disagree if it did not.

## 2. Direction is decided by position, not by "who"/"what"

> **Rule.** If the matched entity phrase occurs **before** the relation phrase, the search
> is **outgoing**. If it occurs **after**, the search is **incoming**.

This single rule produces the correct direction for all twelve specified questions with no
per-question special-casing:

| Question | Entity position | Direction |
|---|---|---|
| Who founded Aether Analytics? | after `founded` | incoming ✓ |
| What does Aether Analytics develop? | before `develops` | outgoing ✓ |
| Who works on OrionEdge? | after `works on` | incoming ✓ |
| Who leads Project Aurora? | after `leads` | incoming ✓ |
| Where is OrionEdge deployed? | after `deployed` | incoming ✓ |
| What is integrated into OrionEdge? | after `integrated into` | incoming ✓ |

### Why not key off "who"/"what"?

That is the obvious shortcut and it is wrong. `Where is OrionEdge deployed?` begins with
"where", which is neither "who" nor "what", so a who/what heuristic has no answer for it
— yet the question plainly needs an **incoming** search on OrionEdge. Worse, a heuristic
keyed on question-word *type* conflates "what is being asked about" with "which way does
the edge point", which are unrelated.

The position rule has no such failure mode, because it reads the actual structure of the
sentence rather than its interrogative word.

### Symmetric override

`collaborates_with` and `partners_with` are symmetric by meaning, so both override to
`both` regardless of position.

## 3. Entity matching

Order of preference (spec §20), with no fuzzy matching and no embeddings:

1. exact canonical name → `entity_match="canonical"`
2. exact alias → `entity_match="alias"`
3. normalised lowercase → `entity_match="canonical"` (still exact after case folding)
4. declared anaphor/description from `data/entities.yaml` → `entity_match="anaphor"`
5. no match → `entity_match="none"`, and the parser does **not** invent an entity

A leading article is stripped when matching (`the company` → `company`), which is what
lets `"Who improved the Entity Resolution Engine?"` resolve. Multi-word relation phrases
are matched before single words, so `"works at"` cannot be shadowed by `"works on"`.

## 4. The synonym table is ambiguous by construction

The specification's table maps individual phrases to more than one relation. Two cases
survive in `RELATION_SYNONYMS` (`src/query.py:35`):

```
collaborates_with ← "works with"
partners_with     ← "works with"
develops          ← "develops", "developed"
works_on          ← "works on"
```

**This is unresolvable without inventing a tie-break rule the specification does not
state.**

### Decision: search all matches and say so

Rather than silently picking one, the parser:

1. collects **every** relation sharing the matched phrase position,
2. searches **all** of them,
3. sets `ambiguous=True`.

The UI then says:

> That phrase maps to more than one relation in the specification's synonym table, so every
> matching relation was searched rather than guessing one.

Silently choosing one would hide a real ambiguity in the spec and produce an answer that
looks confident and is arbitrarily narrow. `AMBIGUOUS_PHRASES` (`src/query.py:74`) names
the four known cases — `works with`, `works on`, `develops`, `developed` — so the warning
does not depend on which relation happens to be listed first in a dict.

Note `"develops"` is genuinely useful in both senses — *"Aether develops OrionEdge"* is
`develops`, while *"Arun develops the ledger"* is `works_on` — so this ambiguity is real
language, not a typo in the spec.

## 5. Connection queries (§22)

```text
How is Arun Mehta connected to OrionEdge?
```

Breadth-first search over edges **in both directions**, `max_depth` = 3, shortest paths
first.

### The search must be undirected

The specification's own worked example requires traversing an edge backwards:

```
Dr. Mira Sen ──leads──▶ Project Aurora ◀──leads── Nila Rao
```

`leads` is stored from the leader *to* the project. An outgoing-only search from Nila Rao
finds Nila's own outgoing edges and cannot reach Aether Analytics via Mira. So `_bfs`
walks `out_edges` **and** `in_edges`.

### Reverse hops are rendered as English

A backwards hop is *not* drawn as `Project Aurora ──leads──▶ Nila Rao`, because that
asserts a relation the graph does not contain. `INVERSE_PHRASES` (`src/query.py:489`)
renders it as `led by`, `founded by`, `developed by`, `used by`, `improved by` — prose that
is true — and the UI adds:

> a ← hop means the stored edge points the other way

### Ordering

BFS yields shortest paths first. The specification's 3-hop worked example is therefore
**not** `paths[0]`: a 2-hop route exists via Arun Mehta. Tests assert membership and
contiguity rather than index 0, because "the spec's path is present" is the real
requirement.

## 6. Answer ordering

Answers are deduplicated to one row per entity, then sorted by `(document_id,
sentence_index)`.

`graph.in_edges(n)` yields predecessors in adjacency-insertion order, which has nothing to
do with the documents. Sorting explicitly means `"Who leads Project Aurora?"` reads
Dr. Mira Sen (`doc_01`) then Nila Rao (`doc_02`) — document order, which is what a reader
expects from a summary of a corpus. The ordering is deterministic across runs.

## 7. Result shape

```python
{
  "answers":          [str],      # deduplicated, document-ordered
  "matched_triples":  [(subject, relation, object)],   # oriented query-entity-first
  "evidence":         [dict],     # sentence + rule_id + confidence + resolutions
  "paths":            [[step]],   # connection queries; step = {from, to, relation, direction}
  "parse":            QueryParse,
  "message":          str,        # only when answers is empty
}
```

### Evidence and the coreference demonstration

`evidence[i]["resolutions"]` is a list of `{from, to}` pairs where the text said a pronoun
or an alias and the graph stores a canonical entity. The UI renders:

> Resolved: "He" → **Arun Mehta**

The UI shows this line **only when `resolutions` is non-empty**. On `"Who founded Aether
Analytics?"` both endpoints were named literally, so the line is omitted — claiming
coreference did work when it did nothing would oversell the pipeline, and on a
case-study screen that is exactly the kind of claim a reviewer will test.

## 8. Silent bugs worth remembering

Three defects here produced *wrong answers without raising*, which is the dangerous class:

1. **`in_edges`/`out_edges` endpoints swapped.** `out_edges(n)` yields `(n, v)` so the
   answer is `v`; `in_edges(n)` yields `(u, n)` so the answer is `u`. Having these
   backwards made every single-answer query return the queried entity itself.

2. **A shared prefix-step object in BFS.** One `dict` was reused by every path descending
   through a given edge, so popping the direction flag on the first path left later paths
   labelling the same hop backwards. Now read with `.get("_forward", True)`.

3. **`zip(a, a[1:], strict=True)` always raises.** Pairwise iteration over a list and its
   own one-shorter tail is unequal *by construction*. `strict=False`, or zip the other
   way.

None of these would have failed a smoke test. Each needed an assertion on the answer.

## 9. Deliberate non-features

- No fuzzy or typo-tolerant matching (`"Aether Analytcs"` will not match).
- No multi-relation conjunction (`"who works at and develops"`).
- No counting, aggregation, or superlative questions (`"how many products..."`).
- No query over relation evidence text.
- An unrecognised query returns an explanatory message and an empty result. It never
  returns a guess.

Each of these would be a new feature with new failure modes, and the specification asks for
twelve specific templates, which are implemented and tested.