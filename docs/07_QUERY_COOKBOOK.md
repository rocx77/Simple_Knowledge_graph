# 07 — Query cookbook

A set of verified questions that exercise the pipeline end to end, with a
focus on complex multi-hop graph traversal rather than single-triple lookups.
They are asserted in `tests/test_query_cookbook.py`.

| Question | Expected answer |
|---|---|
| Who made Mario? | Shigeru Miyamoto, Nintendo |
| Who created Mario? | Shigeru Miyamoto, Nintendo |
| How is Helios Bank connected to OpenAI? | OpenAI |
| How is Helios Bank connected to Netscape? | Netscape |
| How is OrionEdge connected to Linux Foundation? | Linux Foundation |
| How is GraphMatch connected to Mozilla? | Mozilla |
| How is Nila Rao connected to GraphMatch? | GraphMatch |
| How is Priya Nair connected to Kyoto Animation? | (no connection) |

Notes:

- `Who made Mario?` / `Who created Mario?` exercise synonym robustness for
  `develops` (the query parser now treats `made` and `created` as synonyms).
- The `How is ... connected to ...?` questions are breadth-first multi-hop
  traversals across the 14 weakly connected components of the graph.
- A `(no connection)` answer is a real, correct result: the entities exist
  but no path up to `KG_MAX_PATH_LENGTH` joins them.
