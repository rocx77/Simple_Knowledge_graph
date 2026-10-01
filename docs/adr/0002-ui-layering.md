# ADR 0002 — UI layering and layout

- **Status:** accepted
- **Date:** 2026-10-02
- **Charter invariants:** I1 (no UI library reachable from `src/`), I5, I9
- **Spec:** §15–33, §42

## Context

The charter forbids `streamlit`, `pyvis` and `app` from being imported by anything under
`src/`, and §42 requires that `python -m src.pipeline` produce the artifacts with no UI in
the picture. Separately, the guidelines *referenced* `tests/test_architecture.py` as the
enforcement point for I1 — **and that file did not exist**. Six invariant test files were
named in the charter; none were present.

## Decision

### 1. `app.py` at the repo root, `ui/` beside it

```
app.py              layout and widget state only
ui/theme.py         palette + CSS, no Streamlit import (so it is unit-testable)
ui/graph.py         offline PyVis renderer (see ADR 0001)
ui/data.py          cached loaders; the only module importing both streamlit and src/
src/                pipeline; imports neither
```

`app.py` holds no business logic and `ui/data.py` is the single seam that touches
`streamlit` *and* `src/`. Adding a second such seam would make the invariant harder to
enforce, not easier.

### 2. Detect forbidden imports by parsing the AST, not by grepping

`src/` legitimately mentions Streamlit and PyVis **in prose** — module docstrings and the
logging configuration. A text search produces false positives, and the natural response to
a false positive is to silence the check, which destroys the invariant it was meant to
protect. `tests/test_architecture.py` walks `ast.Import` / `ast.ImportFrom` nodes instead,
so only real imports are flagged.

### 3. `st.iframe` for the graph, `st.html` for CSS

The graph needs real JavaScript, so it goes in an iframe via `st.iframe(html_string)`.
`st.components.v1.html` is deprecated and is not used. CSS is injected with
`st.html`. Note the asymmetry this creates, which the screenshot harness depends on:
**Streamlit's own chrome lives in the main frame; only the graph is inside an iframe.**

### 4. Cache models as resources, artifacts as data

| Object | Cache | Why |
|---|---|---|
| `KnowledgeGraphPipeline` (spaCy + FastCoref models) | `@st.cache_resource` | large, unserialisable, should be shared not copied |
| `KnowledgeBase` (graph, triples, report) | `@st.cache_data` | plain values, hashable, cheap to re-derive |
| `EntityRegistry` | `@st.cache_resource` | immutable after load |

Rebuilds bump a `rebuild_token` that participates in the cache key rather than calling
`st.cache_data.clear()`, which is narrower and does not evict unrelated entries. Rebuilding
is always an explicit button press — §41 requires that querying never re-runs NLP.

### 5. Confidence is labelled as a label everywhere

I7 requires confidence to be stated as a rule label, not a probability. It appears on every
evidence card and in every edge tooltip.

## Consequences

Good:
- `python -m src.pipeline` still runs headless; asserted by test, not by convention.
- The offline guarantee (§ ADR 0001) and the no-UI-imports guarantee are both executable.
- The layout matches the specification's required structure (§28) closely.

Costs, accepted:
- `st.set_page_config` and Streamlit's global singleton make in-process `AppTest` runs
  order-sensitive. See the next ADR-worthy item below.
- I5 ("one failing sentence never aborts the pipeline") has **no test yet**. The name
  `tests/test_pipeline.py` is still unwritten, as are `test_acceptance.py` (I2),
  `test_coreference.py` (I3), `test_serialization.py` (I4) and `test_relations.py` (I8).
  Their coverage partly exists under other filenames.

## The load-bearing sys.modules detail

`tests/test_architecture.py::test_no_transitive_import_of_ui_libraries` evicts every
`streamlit*` module from `sys.modules` and restores them afterwards.

Evicting only the top-level name is **not** sufficient. Streamlit caches a process-wide
`DeltaGeneratorSingleton` in a *submodule*, so a partial eviction leaves a stale instance
behind and every later `import streamlit` fails with:

```
RuntimeError: DeltaGeneratorSingleton instance already exists!
```

This was observed: it made all 12 `AppTest` tests in `tests/test_ui.py` error out — but
**only** when `test_architecture.py` ran first. Running `tests/test_ui.py` alone passed
47/47. The test must save and restore the whole `streamlit*` subtree. Do not simplify it.

## Follow-ups

1. Create the five missing invariant test files named in the charter, **moving** the
   relevant assertions into them rather than duplicating coverage under a second name.
2. Write I5 for real: nothing currently proves that a raise inside a per-document stage
   degrades to a warning. This is a behavioural gap, not a naming one.
3. Add a `README.md` (§39) — still unwritten.