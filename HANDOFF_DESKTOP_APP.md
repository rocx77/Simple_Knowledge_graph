# Handoff: wrapping the Knowledge Graph Explorer as a desktop app + screenshot harness

Written against the verified state of this repo. Everything below was checked on this
machine; nothing is aspirational.

---

## 1. What exists right now (verified)

| Thing | Detail |
|---|---|
| Repo root | `C:\CODE\NLP_knowledge_graph_Case_Study` |
| UI entry point | `app.py` (repo root — do not move it into `src/`) |
| Launch command | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Currently running | `http://localhost:8511` → HTTP 200, started headless |
| Readiness probe | `/_stcore/health` → `200 ok` (verified against the live server) |
| UI modules | `ui/theme.py` (palette + CSS), `ui/graph.py` (offline PyVis renderer), `ui/data.py` (cached loaders) |
| Artifacts | `artifacts/graph.json`, `triples.csv`, `entities.json`, `pipeline_report.json` |
| Tests | `164 passed` via `.\.venv\Scripts\python.exe -m pytest tests -q` |
| Headless UI tests | `tests/test_ui.py` uses `st.testing.v1.AppTest`, with `APP = "../app.py"` |
| Lint | `ruff check src tests ui app.py` → clean |
| Graph payload | ~719 KB self-contained HTML, vis.js inlined, **zero external references** |

The app auto-builds the artifacts on first launch if they are missing, so a fresh clone
works with no setup step.

### Environment already confirmed

- **WebView2 runtime present** — `154.0.4258.48` at
  `HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}`.
  A `pywebview` shell will use it (`edgechromium` backend) with **no** extra system install.
- Python `3.14.3`, Streamlit `1.64.0`, pyvis `0.3.2`, networkx `3.7`.
- `PIL` is installed. **`pywebview`, `playwright` and `pywin32` are NOT installed yet.**

---

## 2. What you are building

Two separable pieces. Build and verify them in this order — the screenshot harness is
useful on its own, before any desktop shell exists.

1. **Desktop shell** — a native window showing the Streamlit app, owning the server
   subprocess lifecycle so the user never types a command.
2. **Screenshot harness** — a script that drives the app in a real browser, captures PNGs
   of specific states, so the UI can be reviewed without a human clicking through it.

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ desktop.py  (pywebview)     │        │ tools/shoot.py  (Playwright)│
│  spawns streamlit ──────────┼───URL──┼──▶ drives http://127.0.0.x  │
│  native window, owns kill   │        │    captures PNGs to shots/  │
└─────────────────────────────┘        └──────────────────────────────┘
```

The shell and the harness both talk to the app over HTTP only. They do not import
`app.py`, so neither can accidentally pull Streamlit into the pipeline path.

---

## 3. Step 1 — the desktop shell

### Install

```powershell
.\.venv\Scripts\python.exe -m pip install pywebview
```

`pywebview` pulls in pythonnet/cef on some platforms; on Windows it uses the installed
WebView2 runtime and needs no compiler.

### `desktop.py`

```python
"""Native window over the local Streamlit app.

Owns the server subprocess: starts it, waits for readiness, opens the window, and kills
the server on exit so no orphaned python processes accumulate between demo runs.
"""

from __future__ import annotations

import multiprocessing
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
READY_TIMEOUT = 120.0  # generous: first launch also runs the NLP pipeline


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_healthy(port: int, process: subprocess.Popen, timeout: float) -> None:
    """Poll Streamlit's own health endpoint.

    ``/_stcore/health`` is the correct readiness signal: hitting ``/`` returns 200 while
    the websocket backend is still starting, which yields a window that renders blank.
    """
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"streamlit exited early with code {process.returncode}; "
                "run `python -m streamlit run app.py` directly to see the traceback."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    raise TimeoutError(f"streamlit was not healthy on port {port} within {timeout}s")


def main() -> int:
    multiprocessing.freeze_support()  # required when frozen by PyInstaller
    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
    )
    try:
        print(f"starting streamlit on 127.0.0.1:{port} …", flush=True)
        wait_until_healthy(port, server, READY_TIMEOUT)
        print(f"ready: http://127.0.0.1:{port}", flush=True)
        webview.create_window(
            "Knowledge Graph Explorer",
            f"http://127.0.0.1:{port}",
            width=1600,
            height=1200,
        )
        webview.start()
    finally:
        # Always tear the server down, including on a window close or Ctrl+C.
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it:

```powershell
.\.venv\Scripts\python.exe desktop.py
```

### Desktop-shell notes worth carrying forward

- **Poll `/_stcore/health`, not `/`.** Already handled above.
- **Use a free ephemeral port.** A fixed port collides with a dev server you left
  running. If you must pin a port for a demo, handle the `address already in use` case.
- **`multiprocessing.freeze_support()`** before anything else, or a PyInstaller build
  will re-spawn on Windows.
- pywebview gives you **no screenshot API** on the WebView2 backend. Do not design the
  screenshot feature on top of it — that is what Step 2 is for.
- If you later freeze with PyInstaller, add `--hidden-import streamlit` and ship
  `en_core_web_sm` plus the FastCoref weights. That is a separate, larger task.

---

## 4. Step 2 — the screenshot harness

### Why Playwright rather than a window grab

A window grab needs `pywin32`, fights DPI scaling, and captures whatever happens to be
visible. Playwright gives deterministic full-page PNGs, viewport control, and the ability
to *drive* the app into a specific state (run a query, tick the debug checkbox, filter to
one entity type) before capturing. That is what "review the webpage" actually needs.

The tradeoff is a one-time ~150 MB Chromium download. That is a **dev-only** dependency:
keep it out of `requirements.txt` so the runtime stays offline-clean.

### Install

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe -m playwright install chromium
```

### Stable selectors already in `app.py`

These are real hooks you can rely on, so the script does not depend on DOM structure:

| Target | Selector |
|---|---|
| Query box | `get_by_placeholder("Who founded Aether Analytics?")` |
| Execute button | `get_by_role("button", name="Execute")` |
| Example query dropdown | first `selectbox` (option `"Who leads Project Aurora?"`) |
| Entity-type filter | first `multiselect` (e.g. set to `["PERSON"]`) |
| Relation-type filter | second `multiselect` |
| Search entity | `get_by_placeholder("Nila Rao")` |
| Debug checkbox | `get_by_label("Show NLP pipeline details")` |
| Rebuild button | `get_by_role("button", name="Rebuild knowledge graph")` |
| The graph itself | inside `iframe` → `#mynetwork canvas` |

> **Unverified selector — check this before trusting step `06`.** Everything above is
> either a label, a placeholder, a role or an id that exists in `app.py` and I checked.
> The entity-type **multiselect** is the exception: Streamlit's `data-testid` values are
> emitted by its compiled frontend, not by Python source, so I could not confirm the
> `stMultiSelect` testid from this repo and deliberately did not guess one. The snippet
> above uses visible text instead. If that proves flaky, open DevTools, right-click the
> dropdown, copy the selector, and paste it in.
>
> You do **not** need the browser to prove filtering works: `AppTest` already does, in
> `tests/test_ui.py::TestApp::test_entity_filter_updates_the_graph`
> (`run.multiselect[0].set_value(["PERSON"])`, then asserts `3 nodes`). The screenshot is
> only for reviewing how the filtered graph *looks*. If the locator proves painful, leave
> that one shot out — nothing else depends on it.

**Critical Streamlit detail:** Streamlit's own chrome (query box, KPIs, tables) lives in
the **main frame**. Only the graph is inside an iframe, because `app.py` renders it with
`st.iframe(...)`. So Playwright reaches the controls directly and the graph through
`frame_locator("iframe")`.

### `tools/shoot.py`

```python
"""Capture review screenshots of the Streamlit app.

Assumes a server is already listening (see tools/serve.py, or desktop.py).
Every state is captured full-page at 1600x1200 so nothing is cropped.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from playwright.sync_api import Page, expect, sync_playwright

# Streamlit reruns asynchronously; after an interaction the DOM is briefly stale.
SETTLE_MS = 1200
# vis.js forceAtlas2Based stabilises over ~220 iterations; wait for real canvas pixels.
GRAPH_SETTLE_MS = 3000


def shoot(page: Page, out: pathlib.Path, name: str, settle: int = SETTLE_MS) -> None:
    page.wait_for_timeout(settle)
    target = out / f"{name}.png"
    page.screenshot(path=str(target), full_page=True)
    print(f"  saved {target}")


def wait_for_graph(page: Page) -> None:
    """Block until vis.js has actually painted inside the graph iframe.

    Waiting on the iframe element alone is not enough -- it exists before vis.js draws.
    Waiting on the canvas is the signal that means 'the graph is on screen'.
    """
    canvas = page.frame_locator("iframe").locator("#mynetwork canvas")
    expect(canvas).to_be_visible(timeout=30_000)
    page.wait_for_timeout(GRAPH_SETTLE_MS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8511")
    parser.add_argument("--out", default="shots")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(args.url, wait_until="networkidle")

        print("capturing:")
        shoot(page, out, "01_default_graph", settle=2000)
        wait_for_graph(page)
        shoot(page, out, "02_graph_settled")

        # Query: works_on, which is the coreference-dependent one.
        page.get_by_placeholder("Who founded Aether Analytics?").fill(
            "Who works on OrionEdge?"
        )
        page.get_by_role("button", name="Execute").click()
        shoot(page, out, "03_query_with_evidence", settle=2000)

        # Connection query: path rendering.
        page.get_by_placeholder("Who founded Aether Analytics?").fill(
            "How is Nila Rao connected to Aether Analytics?"
        )
        page.get_by_role("button", name="Execute").click()
        shoot(page, out, "04_connection_paths", settle=2000)

        # Search + neighbourhood panel.
        page.get_by_placeholder("Nila Rao").fill("Nila Rao")
        shoot(page, out, "05_entity_search", settle=2000)

        # Filter to PERSON only.  SEE NOTE BELOW before relying on this step.
        page.get_by_placeholder("Nila Rao").fill("")
        page.get_by_text("Entity type", exact=True).click()
        page.get_by_text("PERSON", exact=True).first.click()
        page.keyboard.press("Escape")
        shoot(page, out, "06_filtered_person_only", settle=2500)

        # Debug mode.
        page.get_by_label("Show NLP pipeline details").check()
        shoot(page, out, "07_pipeline_details", settle=2000)

        browser.close()
    print(f"done -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Run it

```powershell
# terminal 1
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8511 --server.headless true

# terminal 2
.\.venv\Scripts\python.exe tools\shoot.py --url http://127.0.0.1:8511 --out shots
```

---

## 5. What to actually review in the screenshots

Mapped to the specification so review is objective rather than taste:

| Shot | Check | Spec |
|---|---|---|
| `01_default_graph` | 4 KPI cards read Documents 5 / Entities 10 / Relations 16 / Graph edges 16 | §16 |
| `02_graph_settled` | Bubble nodes; **one colour per entity type**; arrows on every edge; **relation label on every edge**; hubs (Aether, Aurora) visually larger | §15, §43 |
| `02_graph_settled` | Hover a node → Entity / Type / Aliases / Mention count / Documents. Hover an edge → Relation / Source document / Original sentence / Confidence | §15 |
| `03_query_with_evidence` | Interpretation block shows Entity / Relation / Direction / Intent. Answer card. Evidence quote **with** `Resolved: "He" → Arun Mehta` | §17, §23 |
| `04_connection_paths` | Path renders with `── relation ──▶`; a `←` hop is explained as the stored edge pointing the other way | §22 |
| `05_entity_search` | Searched node highlighted, non-neighbourhood dimmed but **still present**, connections table below | §25, §26 |
| `06_filtered_person_only` | 3 nodes / 3 edges; no dangling edges; "Show full graph" restores | §25 |
| `07_pipeline_details` | Per document: text, sentences, entities, coreference, relations, triples | §24 |

Also confirm, once: **no network access at runtime.** Open DevTools → Network, reload,
and the graph must load with zero external requests. `tests/test_ui.py` asserts this in
code, but it is worth seeing once.

---

## 6. Fragile spots — read before you change anything

1. **`ui/graph.py` depends on pyvis internals.** Charter invariant I10 requires vis.js to
   be embedded, and pyvis `0.3.2` cannot do that itself (`cdn_resources="local"` still
   emits four CDN URLs). `ui/graph.py` inlines vis-network from
   `pyvis/templates/lib/vis-9.1.2/` and drops Bootstrap. **Upgrading pyvis breaks this
   silently-ish**: `pyvis_asset_dir()` raises with an actionable message if the directory
   moved, and `render_graph()` raises `RuntimeError` if any `http(s)` reference survives.
   Both are covered by tests — run them after any pyvis bump.

2. **`app.py` must stay at the repo root.** Charter invariant I1 forbids anything under
   `src/` importing `streamlit`, `pyvis` or `app`. Keep `ui/` and `app.py` outside `src/`.

3. **Never put `playwright` or `pywebview` in `requirements.txt`.** They would weaken the
   offline/no-extra-dependency claim. Put them in a separate `requirements-dev.txt`.

4. **`tests/test_architecture.py::test_no_transitive_import_of_ui_libraries`** evicts every
   `streamlit*` module from `sys.modules` and restores them afterwards. If you ever
   simplify that to `sys.modules.pop("streamlit")`, it will break
   `tests/test_ui.py` with *"DeltaGeneratorSingleton instance already exists!"*. It is
   load-bearing.

5. **First launch is slow** (~10 s) because the app builds the graph if
   `artifacts/graph.json` is absent. `desktop.py` allows 120 s for this reason; do not
   shorten it below ~60 s.

6. **Rebuild button clears `st.cache_data` globally.** Expected, but it means a rebuild
   re-renders every widget. Screenshot scripts should not click it mid-capture.

---

## 7. Definition of done

- [ ] `.\.venv\Scripts\python.exe desktop.py` opens a native window showing the graph, with
      no orphaned python processes left after closing the window.
- [ ] `tools\shoot.py` produces all seven PNGs against a running app.
- [ ] Every row in the §5 review table has been checked against an actual screenshot.
- [ ] DevTools Network panel is empty on reload.
- [ ] `.\.venv\Scripts\python.exe -m pytest tests -q` → `164 passed`.
- [ ] `.\.venv\Scripts\python.exe -m ruff check src tests ui app.py` → clean.
- [ ] `requirements.txt` is unchanged; new desktop/screenshot deps live in
      `requirements-dev.txt`.