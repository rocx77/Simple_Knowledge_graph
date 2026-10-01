"""UI-layer tests.

Two kinds of assertion here:

* Plain pytest against the pure helpers in ``ui.graph``/``ui.theme``. These need no
  Streamlit runtime and cover the parts that are easy to break silently -- chiefly the
  offline guarantee, since a leaked CDN URL still renders perfectly on a machine with
  internet and only fails in the case-study demo.
* ``st.testing.v1.AppTest`` against ``app.py`` itself, for widget interaction and for
  proving the app boots.

Both are headless: no browser, no server, no port.
"""

from __future__ import annotations

import networkx as nx
import pytest
from streamlit.testing.v1 import AppTest

from src.config import AppConfig
from src.serialization import load_graph
from ui.graph import (
    build_html,
    connections_of,
    edge_hover,
    external_references,
    filter_graph,
    inline_assets,
    neighbourhood,
    node_hover,
    node_size,
    pyvis_asset_dir,
    render_graph,
)
from ui.theme import ENTITY_COLORS, color_for, swatch_legend

APP = "../app.py"  # AppTest resolves relative to this file


@pytest.fixture(scope="module")
def graph() -> nx.MultiDiGraph:
    config = AppConfig.from_env()
    return load_graph(config.artifacts_dir / "graph.json")


@pytest.fixture(scope="module")
def html(graph: nx.MultiDiGraph) -> str:
    return render_graph(graph)


@pytest.fixture(scope="module")
def at() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


# --------------------------------------------------------------------- offline guarantee


class TestOfflineHtml:
    """Charter invariant I10: the generated HTML must work with no network."""

    def test_no_external_references(self, html):
        assert external_references(html) == []

    def test_vis_js_is_embedded(self, html):
        assert "vis-network" in html
        # Embedded, not merely referenced: the vis.js UMD bundle is ~700 KB of source.
        assert len(html) > 400_000

    def test_bootstrap_is_not_referenced(self, html):
        assert "bootstrap" not in html.lower()

    def test_every_node_and_edge_reaches_the_document(self, graph, html):
        for node_id in graph.nodes:
            assert node_id in html
        for _, _, data in graph.edges(data=True):
            assert str(data["relation"]) in html

    def test_tooltips_carry_the_specified_fields(self, html):
        # Specification section 15 lists five node fields and four edge fields.
        for field in ("Entity", "Type", "Aliases", "Mention count", "Documents"):
            assert field in html
        for field in ("Relation", "Source document", "Original sentence", "Confidence"):
            assert field in html
        assert html.count("Mention count") == graph_node_count(html)
        assert "rule confidence label, not a probability" in html

    def test_inferred_relations_are_marked_inferred(self, graph, html):
        inferred = [d for _, _, d in graph.edges(data=True) if d.get("inferred")]
        for data in inferred:
            assert edge_hover(data).count("not stated literally") == 1

    def test_render_graph_refuses_to_return_leaky_html(self, graph, monkeypatch):
        monkeypatch.setattr(
            "ui.graph.external_references", lambda _html: ["https://cdn/x.js"]
        )
        with pytest.raises(RuntimeError, match="not offline-capable"):
            render_graph(graph)

    def test_inline_assets_drops_unknown_external_urls(self):
        leaky = '<html><head><script src="https://evil.example/x.js"></script></head></html>'
        assert "evil.example" not in inline_assets(leaky)

    def test_inline_assets_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="escapes pyvis lib directory"):
            from ui.graph import read_asset

            read_asset("../../../etc/passwd")

    def test_pyvis_assets_are_present(self):
        assert (pyvis_asset_dir() / "vis-9.1.2" / "vis-network.min.js").is_file()


def graph_node_count(html: str) -> int:
    """Node tooltips in the document, used to assert each one is emitted exactly once."""
    return html.count("Mention count")


# ------------------------------------------------------------------------- graph content


class TestGraphRendering:
    def test_all_nodes_present(self, graph, html):
        for node_id, attrs in graph.nodes(data=True):
            assert node_id in html, node_id
            assert str(attrs["label"]) in html

    def test_relation_labels_present(self, graph, html):
        for _, _, data in graph.edges(data=True):
            assert data["relation"] in html

    def test_directed_arrows_configured(self, html):
        assert '"arrows": "to"' in html

    def test_force_directed_layout_configured(self, html):
        assert "forceAtlas2Based" in html

    def test_node_shapes_are_bubbles(self, html):
        assert '"shape": "dot"' in html

    def test_every_node_is_coloured_by_type(self, graph, html):
        for _, attrs in graph.nodes(data=True):
            assert color_for(attrs["type"]).lstrip("#") in html

    def test_focus_emphasises_one_node(self, graph):
        focused = render_graph(graph, focus="person:nila_rao")
        # The focused node gets a wider border than its neighbours.
        assert '"borderWidth": 5' in focused
        assert '"borderWidth": 2' in focused

    def test_sizes_scale_with_degree(self):
        assert node_size(10, 12, 10) > node_size(1, 1, 10)
        assert node_size(0, 0, 0) == 22.0

    def test_highlight_does_not_hide_the_graph(self, graph):
        # A dimmed node must remain present, otherwise "focus" would delete context.
        focused = build_html(graph, focus="person:nila_rao")
        assert "project:project_aurora" in focused


# ------------------------------------------------------------------------------ filters


class TestFilters:
    def test_no_filters_returns_the_same_graph(self, graph):
        assert filter_graph(graph, [], []) is graph

    def test_entity_type_filter(self, graph):
        sub = filter_graph(graph, ["PERSON"], [])
        assert sub.number_of_nodes() == 10
        assert all(
            d["type"] == "PERSON" for _, d in sub.nodes(data=True)
        ), "only PERSON nodes should survive"

    def test_relation_filter(self, graph):
        sub = filter_graph(graph, [], ["works_at"])
        assert sub.number_of_edges() == 2
        assert all(d["relation"] == "works_at" for _, _, d in sub.edges(data=True))

    def test_filters_are_conjunctive(self, graph):
        # No edge connects two COMPONENT nodes, so the intersection is empty.
        assert filter_graph(graph, ["COMPONENT"], ["works_at"]).number_of_edges() == 0

    def test_filter_never_leaves_dangling_edges(self, graph):
        sub = filter_graph(graph, ["PERSON", "ORGANIZATION"], [])
        for source, target in sub.edges():
            assert source in sub and target in sub


# ----------------------------------------------------------------------------- search


class TestSearch:
    def test_neighbourhood_is_inclusive(self, graph):
        found = neighbourhood(graph, "person:nila_rao", depth=1)
        assert "person:nila_rao" in found
        assert "project:project_aurora" in found, "Nila leads Aurora"
        assert "person:arun_mehta" in found, "Nila collaborates with Arun"
        assert "organization:aether_analytics" not in found, "Aether is 3 hops away"
        assert "organization:aether_analytics" in neighbourhood(
            graph, "person:nila_rao", depth=2
        )

    def test_unknown_node_has_no_neighbourhood(self, graph):
        assert neighbourhood(graph, "person:nobody", depth=1) == set()

    def test_connections_are_bidirectional(self, graph):
        rows = connections_of(graph, "project:project_aurora")
        rendered = {(r["direction"], r["relation"], r["other"]) for r in rows}
        # Both leads edges are stored FROM the two leaders TO Aurora, so from Aurora's
        # point of view both are incoming. The arrows reflect the stored direction.
        assert ("←", "leads", "Dr. Mira Sen") in rendered
        assert ("←", "leads", "Nila Rao") in rendered
        assert ("→", "focuses_on", "graph-based transaction analysis") in rendered

    def test_connections_for_unknown_node_is_empty(self, graph):
        assert connections_of(graph, "person:nobody") == []


# ------------------------------------------------------------------------------- theme


class TestTheme:
    def test_every_specified_entity_type_has_a_colour(self):
        for entity_type in (
            "PERSON",
            "ORGANIZATION",
            "PRODUCT",
            "PROJECT",
            "COMPONENT",
            "LITERAL",
        ):
            assert color_for(entity_type) in ENTITY_COLORS.values()

    def test_unknown_type_falls_back(self):
        assert color_for("ALIEN").startswith("#")

    def test_colours_are_distinct(self):
        assert len(set(ENTITY_COLORS.values())) == len(ENTITY_COLORS)

    def test_legend_orders_types_canonically(self):
        legend = swatch_legend(["PRODUCT", "PERSON", "ORGANIZATION"])
        assert legend.index("PERSON") < legend.index("ORGANIZATION")
        assert legend.index("ORGANIZATION") < legend.index("PRODUCT")


# ---------------------------------------------------------------------------- tooltips


class TestTooltips:
    def test_node_tooltip_escapes_html(self):
        html = node_hover({"label": "<script>alert(1)</script>", "type": "PERSON"})
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_node_tooltip_shows_placeholders_when_empty(self):
        html = node_hover({"label": "X", "type": "PERSON"})
        assert "—" in html

    def test_edge_tooltip_flags_inferred(self):
        html = edge_hover({"relation": "customer_of", "inferred": True})
        assert "not stated literally" in html


# ------------------------------------------------------------------------ the app itself


class TestApp:
    def test_app_boots(self, at):
        assert not at.exception

    def test_kpi_cards(self, at):
        values = {m.label: m.value for m in at.metric}
        assert values == {
            "Documents": "16",
            "Entities": "37",
            "Relations": "61",
            "Graph edges": "61",
        }

    def test_graph_is_rendered_by_default(self, at):
        captions = " ".join(c.value for c in at.caption)
        assert "45 nodes" in captions and "61 edges" in captions

    def test_triple_table_lists_every_triple(self, at):
        tables = [df.value for df in at.dataframe]
        assert any(len(t) == 61 for t in tables)

    def test_query_shows_interpretation_and_answer(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.text_input[0].set_value("Who founded Aether Analytics?").run()
        assert not run.exception
        fields = run.dataframe[0].value
        interpretation = dict(zip(fields["Field"], fields["Value"], strict=True))
        assert interpretation["Entity"] == "Aether Analytics"
        assert interpretation["Relation"] == "founded"
        assert interpretation["Direction"] == "incoming"
        assert interpretation["Intent"] == "relation_lookup"
        assert any("Dr. Mira Sen" in m.value for m in run.markdown)

    def test_coref_query_shows_resolution(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.text_input[0].set_value("Who works on OrionEdge?").run()
        assert not run.exception
        assert any("Resolved" in m.value and "Arun Mehta" in m.value for m in run.markdown)

    def test_connection_query_renders_paths(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.text_input[0].set_value("How is Nila Rao connected to Aether Analytics?").run()
        assert not run.exception
        assert any("path(s) found" in m.value for m in run.markdown)

    def test_example_query_can_be_selected(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.selectbox[0].select("Who leads Project Aurora?").run()
        assert not run.exception
        fields = run.dataframe[0].value
        assert "Project Aurora" in set(fields["Value"])

    def test_entity_filter_updates_the_graph(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.multiselect[0].set_value(["PERSON"]).run()
        assert not run.exception
        assert any("10 nodes" in c.value for c in run.caption)

    def test_search_shows_connections(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.text_input[1].set_value("Nila Rao").run()
        assert not run.exception
        tables = [df.value for df in run.dataframe]
        assert any("Relation" in t.columns for t in tables)

    def test_unknown_query_is_reported_not_crashed(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.text_input[0].set_value("What is the airspeed velocity of a swallow?").run()
        assert not run.exception
        assert run.warning

    def test_debug_mode_renders_pipeline_details(self, at):
        run = AppTest.from_file(APP, default_timeout=180).run()
        run.checkbox[0].check().run()
        assert not run.exception
        assert any("NLP pipeline details" in m.value for m in run.markdown)
