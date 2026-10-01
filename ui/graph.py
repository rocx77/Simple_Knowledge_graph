"""Offline PyVis rendering.

Charter invariant I1 forbids ``src/`` from importing pyvis, and invariant I10 requires
the generated HTML to embed vis.js so the graph works with no network. PyVis 0.3.2 does
not satisfy I10 on its own: even with ``cdn_resources="local"`` it emits four CDN URLs
(vis-network js + css, Bootstrap js + css), because ``local`` only affects its own
``utils.js`` binding.

So the PyVis output is post-processed here: vis-network is inlined from the copy PyVis
already ships inside its package, and the two Bootstrap tags are dropped. Streamlit
supplies the page chrome, so Bootstrap is not needed -- but its classes do appear in the
PyVis template, so a small replacement stylesheet keeps the canvas centred and
full-bleed. Everything below the ``</style>`` guard is plain string work, which is why
this module can be tested without a browser.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from ui.theme import ENTITY_COLORS, INFERRED_RELATION_COLOR, RELATION_COLOR, color_for

VIS_VERSION = "9.1.2"

#: How each vis.js asset URL is satisfied offline. ``None`` means "drop the tag".
#: The directory is ``vis-<version>``; pyvis ships the un-minified ``.css`` name.
_ASSET_RULES: dict[str, tuple[tuple[str, str] | None, str]] = {
    "vis-network.min.js": ((f"vis-{VIS_VERSION}", "vis-network.min.js"), "script"),
    "vis-network.min.css": ((f"vis-{VIS_VERSION}", "vis-network.css"), "style"),
    "utils.js": (("bindings", "utils.js"), "script"),
    # Bootstrap is deliberately not vendored: it is ~200 KB of page furniture that
    # Streamlit already provides, and it is the only one of the four with no local copy.
    "bootstrap.min.css": (None, "drop"),
    "bootstrap.bundle.min.js": (None, "drop"),
}

_LINK_TAG = re.compile(r'<link[^>]*?href="(?P<url>[^"]+)"[^>]*?>', re.DOTALL)
_SCRIPT_TAG = re.compile(
    r'<script[^>]*?src="(?P<url>[^"]+)"[^>]*?>\s*?</script>', re.DOTALL
)

#: Minimal stand-in for the Bootstrap rules the PyVis template actually relies on.
_CHROME_CSS = """
html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.container-fluid { width: 100%; padding: 0; margin: 0; }
.row { display: flex; flex-wrap: wrap; margin: 0; }
.col-md-12 { flex: 0 0 100%; max-width: 100%; padding: 0; }
#mynetwork { width: 100%; border: none; }
"""


def pyvis_asset_dir() -> Path:
    """Directory holding the vis.js assets PyVis ships.

    Resolved from the installed package rather than hard-coded, so it keeps working in a
    venv. Raises with an actionable message rather than a bare FileNotFoundError, because
    the failure mode is a silently non-interactive graph.
    """
    import pyvis

    candidate = Path(pyvis.__file__).parent / "templates" / "lib"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"pyvis assets not found at {candidate}. The installed pyvis does not ship "
            "templates/lib; install pyvis==0.3.2 or update ui.graph._ASSET_RULES."
        )
    return candidate


def read_asset(relative: str) -> str:
    """Read one vendored asset, guarded against path traversal out of the lib dir."""
    root = pyvis_asset_dir().resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"asset path escapes pyvis lib directory: {relative}")
    if not target.is_file():
        raise FileNotFoundError(f"pyvis asset missing: {target}")
    return target.read_text(encoding="utf-8")


def _embed(url: str) -> str | None:
    """Replacement markup for an asset URL, or ``None`` to drop the tag."""
    basename = url.rsplit("/", 1)[-1].split("?")[0]
    rule = _ASSET_RULES.get(basename)
    if rule is None:
        return None if "http" in url else f"__KEEP__{url}"
    location, kind = rule
    if location is None:
        return ""
    content = read_asset("/".join(location))
    if kind == "drop":
        return ""
    if kind == "style":
        return f"<style>\n{content}\n</style>"
    # A literal </script> inside a string literal would close the tag early.
    content = content.replace("</script", r"<\/script")
    return f"<script>\n{content}\n</script>"


def inline_assets(html: str) -> str:
    """Replace PyVis' CDN references with inline content.

    Any external ``src``/``href`` that survives this function is a bug, and
    ``tests/test_ui.py`` asserts that none does.
    """
    html = _LINK_TAG.sub(lambda m: _embed(m.group("url")) or "", html)
    html = _SCRIPT_TAG.sub(lambda m: _embed(m.group("url")) or "", html)
    html = html.replace("__KEEP__", "")
    head = f"<style>{_CHROME_CSS}</style>"
    return html.replace("<head>", f"<head>{head}", 1)


def external_references(html: str) -> list[str]:
    """Every remaining absolute URL in a ``src``/``href`` attribute."""
    return re.findall(r'(?:src|href)="(https?://[^"]+)"', html)


# --------------------------------------------------------------------------- styling


def node_hover(attrs: dict) -> str:
    """Tooltip for a node: Entity / Type / Aliases / Mention count / Documents."""
    aliases = ", ".join(attrs.get("aliases") or []) or "—"
    documents = ", ".join(attrs.get("documents") or []) or "—"
    fields = [
        ("Entity", attrs.get("label", "")),
        ("Type", attrs.get("type", "")),
        ("Aliases", aliases),
        ("Mention count", attrs.get("mention_count", 0)),
        ("Documents", documents),
    ]
    return _tooltip(fields)


def edge_hover(attrs: dict) -> str:
    """Tooltip for an edge: Relation / Source document / Original sentence / Confidence."""
    fields = [
        ("Relation", attrs.get("relation", "")),
        ("Source document", attrs.get("document_id", "")),
        ("Original sentence", attrs.get("sentence", "")),
        ("Confidence", _confidence_text(attrs.get("confidence"))),
    ]
    if attrs.get("inferred"):
        fields.append(("Inferred", "yes — not stated literally in the text"))
    return _tooltip(fields)


def _confidence_text(value: object) -> str:
    """Confidence is a rule label, never a probability (spec section 38)."""
    if value is None:
        return "—"
    return f"{float(value):.2f} (rule confidence label, not a probability)"


def _tooltip(fields: Iterable[tuple[str, object]]) -> str:
    rows = "".join(
        f"<tr><td style='padding:1px 8px 1px 0;opacity:.65'>{html_lib.escape(str(k))}</td>"
        f"<td style='padding:1px 0'><b>{html_lib.escape(str(v))}</b></td></tr>"
        for k, v in fields
    )
    return f"<table style='font-family:system-ui;font-size:12px'>{rows}</table>"


def node_size(degree: int, mention_count: int, largest_degree: int) -> float:
    """Bubble radius from degree, nudged by mention count.

    Central entities are meant to read as visually prominent (spec section 43), so the
    floor is not tiny and the top of the range is generous.
    """
    if largest_degree <= 0:
        return 22.0
    share = degree / largest_degree
    prominence = min(1.0, (mention_count or 0) / 12)
    return 18.0 + 34.0 * (0.72 * share + 0.28 * prominence)


def highlight_color(entity_type: str) -> str:
    """Brighter border colour used for a searched/highlighted node."""
    return color_for(entity_type)


# --------------------------------------------------------------------------- filters


def filter_graph(
    graph: nx.MultiDiGraph,
    entity_types: Sequence[str] | None = None,
    relations: Sequence[str] | None = None,
) -> nx.MultiDiGraph:
    """Return the subgraph induced by the selected entity types and relations.

    Both filters are conjunctive. An edge survives only if both endpoints survive, so the
    result is a real induced subgraph rather than a set of dangling arrows.
    """
    if not entity_types and not relations:
        return graph

    kept_types = {t.upper() for t in entity_types} if entity_types else None
    nodes = {
        n
        for n, d in graph.nodes(data=True)
        if kept_types is None or str(d.get("type", "")).upper() in kept_types
    }

    sub = graph.subgraph(nodes).copy()
    if relations:
        wanted = set(relations)
        sub.remove_edges_from(
            [(u, v, k) for u, v, k, d in sub.edges(keys=True, data=True) if d.get("relation") not in wanted]
        )
    return sub


def neighbourhood(graph: nx.MultiDiGraph, node_id: str, depth: int = 1) -> set[str]:
    """Node ids within ``depth`` hops of ``node_id``, inclusive."""
    if node_id not in graph:
        return set()
    undirected = graph.to_undirected(as_view=True)
    return {n for n, d in nx.single_source_shortest_path_length(undirected, node_id, cutoff=depth).items()}


def connections_of(graph: nx.MultiDiGraph, node_id: str) -> list[dict]:
    """Labeled 1-hop connections for the search panel (spec section 26)."""
    if node_id not in graph:
        return []
    label = graph.nodes[node_id].get("label", node_id)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for source, target, data in graph.out_edges(node_id, data=True):
        key = (source, str(data.get("relation")), target)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "direction": "→",
                "relation": data.get("relation"),
                "other": graph.nodes[target].get("label", target),
                "via": label,
                "confidence": data.get("confidence"),
            }
        )
    for source, target, data in graph.in_edges(node_id, data=True):
        key = (source, str(data.get("relation")), target)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "direction": "←",
                "relation": data.get("relation"),
                "other": graph.nodes[source].get("label", source),
                "via": label,
                "confidence": data.get("confidence"),
            }
        )
    return sorted(rows, key=lambda r: (str(r["relation"]), str(r["other"])))


# ------------------------------------------------------------------------- rendering


def build_html(
    graph: nx.MultiDiGraph,
    *,
    height: int = 720,
    focus: str | None = None,
    title: str = "Knowledge Graph",
) -> str:
    """Render the graph to a self-contained, offline HTML document."""
    largest = max((d.get("degree", 1) for _, d in graph.nodes(data=True)), default=1)
    focused = neighbourhood(graph, focus, depth=1) if focus else set()

    network = Network(
        directed=True,
        cdn_resources="local",
        height=f"{height}px",
        width="100%",
        bgcolor="rgba(0,0,0,0)",
        font_color=False,
        notebook=False,
    )

    for node_id, attrs in graph.nodes(data=True):
        entity_type = str(attrs.get("type", "")).upper()
        is_focus = node_id == focus
        color = color_for(entity_type)
        network.add_node(
            node_id,
            label=attrs.get("label", node_id),
            title=node_hover(attrs),
            shape="dot",
            size=node_size(attrs.get("degree", 0), attrs.get("mention_count", 0), largest)
            + (8 if is_focus else 0),
            color={
                "background": color,
                "border": highlight_color(entity_type) if is_focus else "#FFFFFF",
                "highlight": {"background": color, "border": "#111111"},
                "hover": {"background": color, "border": "#111111"},
            },
            borderWidth=5 if is_focus else 2,
            mass=2.5 + (attrs.get("degree", 0) / max(largest, 1)),
            # A non-focus node inside a focused 1-hop neighbourhood is dimmed rather than
            # removed, so the search result keeps its surrounding context visible.
            opacity=0.25 if focused and node_id not in focused else 1.0,
            font={"size": 15 if is_focus else 12, "color": "#FFFFFF", "strokeWidth": 3},
        )

    seen: set[tuple[str, str, str]] = set()
    for source, target, data in graph.edges(data=True):
        relation = str(data.get("relation", ""))
        key = (source, relation, target)
        if key in seen:
            continue
        seen.add(key)
        inferred = bool(data.get("inferred"))
        network.add_edge(
            source,
            target,
            label=relation,
            title=edge_hover(data),
            arrows="to",
            color={
                "color": INFERRED_RELATION_COLOR if inferred else RELATION_COLOR,
                "opacity": 0.25 if focused and source not in focused and target not in focused else 1.0,
            },
            dashes=inferred,
            width=2,
            font={"size": 10, "align": "middle", "strokeWidth": 3, "vadjust": 8},
        )

    network.heading = title
    # forceAtlas2Based is what makes this read as a knowledge network rather than a
    # hairball: hubs drift outward and leaves cluster in. pyvis 0.3.2 calls
    # str.replace() on the options argument, so it must be a JSON string, not a dict.
    network.set_options(
        json.dumps(
            {
                "layout": {
                    "improvedLayout": True,
                    "physics": {
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {
                            "gravitationalConstant": -58,
                            "centralGravity": 0.012,
                            "springLength": 165,
                            "springConstant": 0.05,
                            "damping": 0.6,
                            "avoidOverlap": 0.75,
                        },
                        "stabilization": {"iterations": 220, "fit": True},
                        "minVelocity": 0.6,
                        "maxVelocity": 40.0,
                    },
                },
                "interaction": {
                    "hover": True,
                    "multiselect": True,
                    "navigationButtons": True,
                    "keyboard": False,
                    "dragNodes": True,
                    "zoomView": True,
                    "hideEdgesOnDrag": False,
                },
                "nodes": {"borderWidth": 2, "borderWidthSelected": 4},
                "edges": {
                    "smooth": {"enabled": True, "type": "dynamic", "roundness": 0.4},
                    "font": {"size": 10, "align": "middle"},
                },
            }
        )
    )
    return inline_assets(network.generate_html(notebook=False))


def render_graph(
    graph: nx.MultiDiGraph,
    *,
    height: int = 720,
    focus: str | None = None,
    title: str = "Knowledge Graph",
) -> str:
    """Build HTML and fail loudly if it is not offline-capable."""
    html = build_html(graph, height=height, focus=focus, title=title)
    leaks = external_references(html)
    if leaks:
        raise RuntimeError(
            "generated graph HTML is not offline-capable; external references: "
            + ", ".join(sorted(set(leaks))[:5])
        )
    return html


__all__ = [
    "ENTITY_COLORS",
    "build_html",
    "connections_of",
    "edge_hover",
    "external_references",
    "filter_graph",
    "inline_assets",
    "neighbourhood",
    "node_hover",
    "pyvis_asset_dir",
    "read_asset",
    "render_graph",
]
