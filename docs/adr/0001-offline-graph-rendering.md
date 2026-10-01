# ADR 0001 — Offline graph rendering

- **Status:** accepted
- **Date:** 2026-10-02
- **Charter invariants:** I1 (no UI library reachable from `src/`), I10 (vis.js embedded; the UI works with no network)
- **Spec:** §15 (interactive graph UI, "Use PyVis"), §4 (no network access at runtime)

## Context

Three requirements had to be satisfied at once:

- **I1** — no module under `src/` may import `streamlit`, `pyvis`, or `app`.
- **I10** — vis.js must be embedded in the generated HTML; the UI must work with no network.
- **§15** — use PyVis for visualization, with `app.py` at the repository root.

At first these looked contradictory, and I recorded them as a blocker requiring a docs
amendment. **That reading was wrong** and is corrected here: I1 is scoped to `src/`, and
`app.py` is at the repo root, so PyVis in the UI is already permitted. No amendment is
needed.

The real conflict is between **I10** and PyVis itself.

## The problem

PyVis 0.3.2 with `cdn_resources="local"` **still emits four CDN URLs** in its output:

| URL | Locally available? |
|---|---|
| `vis-network.min.js` | yes — `templates/lib/vis-9.1.2/vis-network.min.js` (458 KB) |
| `vis-network.min.css` | yes — `templates/lib/vis-9.1.2/vis-network.css` (215 KB) |
| `bootstrap.bundle.min.js` | **no** |
| `bootstrap.min.css` | **no** |

`local` only affects PyVis's own `lib/bindings/utils.js`. The generated page is 4 KB of
markup pointing at cdnjs and jsdelivr, so on a machine without internet the graph is
inert — a blank panel. This violates I10 and the §4 ban on runtime network access.

Two further details make the fix less trivial than a string replacement:

- **The CSS filename differs between URL and disk.** The CDN URL is
  `vis-network.min.css`; the shipped file is `vis-network.css`. A naive basename match
  misses it and the graph renders unstyled but *apparently* functional.
- **PyVis ships two vis versions.** `templates/lib/` contains both `vis-9.0.4` and
  `vis-9.1.2`, and they differ substantially (`875 KB` vs `458 KB` for the `.js`). Inlining
  the wrong one produces a page that loads and then breaks at runtime. `VIS_VERSION` is
  pinned explicitly for this reason rather than globbed.

## Options considered

**A. PyVis + inline its bundled vis.js** *(chosen)*
Build the HTML with PyVis as §15 asks, then rewrite the four references: inline vis-network
from PyVis's own package directory, drop the two Bootstrap tags. Satisfies §15, I1 and
I10 simultaneously.

**B. Hand-rolled offline HTML in `src/visualization.py`**
Vendor `vis-network.min.js` + `.css` into the repo and inline them from a `src/` module,
which is the literal wording of I10's "enforced by `src/visualization.py`". Most robust,
but drops PyVis (contradicting §15) and commits ~670 KB of vendored JS.

**C. PyVis as-is, accept the CDN**
Simplest code. Rejected: the graph only renders with internet, which I believe invalidates
the case study.

## Decision

**Option A**, in `ui/graph.py`.

Bootstrap is dropped rather than vendored because it is ~200 KB of page furniture that
Streamlit already provides. Its classes do appear in the PyVis template, so a small
replacement stylesheet (`_CHROME_CSS`) keeps the canvas centred and full-bleed.

The graph renderer lives in `ui/graph.py`, not `src/visualization.py`. I10 names
`src/visualization.py`, but a module there could not import PyVis without breaking I1, so
placing the renderer under `ui/` is the only position satisfying both invariants. This is
a deliberate, documented deviation from I10's *enforcement location* while honouring its
*requirement*.

## Consequences

Good:
- Generated HTML is ~719 KB, self-contained, zero external references — verified, not assumed.
- The specification's PyVis requirement is met literally.
- `src/` remains clean and the pipeline still runs headless.

Costs, accepted:
- **Coupled to PyVis internals.** A PyVis upgrade can break asset resolution. Mitigated
  three ways: `pyvis_asset_dir()` raises with an actionable message if `templates/lib`
  moves; `render_graph()` raises `RuntimeError` if any `http(s)` reference survives; and
  `tests/test_ui.py::TestOfflineHtml` asserts zero external references. Run those tests
  after any PyVis bump.
- A dependency bump can fail at runtime rather than at import.

## Verification

```python
from src.serialization import load_graph
from ui.graph import render_graph, external_references
html = render_graph(load_graph("artifacts/graph.json"))
assert external_references(html) == []
```

Also checked by hand: the output contains `nodes = new vis.DataSet([...])`,
`edges = new vis.DataSet([...])`, `forceAtlas2Based`, and balanced `<script>` tags. The
minified vis.js blob contains the literal `<script` and is checked for `</script`, which is
escaped to `<\/script` so the blob cannot break out of its own tag.