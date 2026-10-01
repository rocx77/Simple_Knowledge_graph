"""Knowledge Graph Explorer -- Streamlit entry point (specification sections 15-33).

Run with:

    streamlit run app.py

This module is deliberately thin. It owns layout and widget state only; the graph
rendering lives in ``ui.graph``, the cached loaders in ``ui.data``, and every NLP concern
in ``src/``. The pipeline never imports this file (charter invariant I1), so
``python -m src.pipeline`` still produces the artifacts with no UI in the picture.
"""

from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from src.query import QueryEngine
from ui.data import (
    KnowledgeBase,
    artifacts_present,
    get_registry,
    load_knowledge_base,
    rebuild,
)
from ui.graph import connections_of, filter_graph, render_graph
from ui.theme import PAGE_CSS, swatch_legend

st.set_page_config(
    page_title="Knowledge Graph Explorer",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUERIES = [
    "Who founded Aether Analytics?",
    "What does Aether Analytics develop?",
    "Who works on OrionEdge?",
    "Who leads Project Aurora?",
    "Where is OrionEdge deployed?",
    "Who uses OrionEdge?",
    "How is Nila Rao connected to Aether Analytics?",
]


# --------------------------------------------------------------------------- helpers


def kpi_row(kb: KnowledgeBase) -> None:
    """Four cards, in the order specification section 16 asks for."""
    with st.container(horizontal=True):
        st.metric("Documents", kb.documents, border=True)
        st.metric("Entities", kb.canonical_entities, border=True)
        st.metric("Relations", len(kb.triples), border=True)
        st.metric("Graph edges", kb.edge_count, border=True)


def interpretation_panel(result) -> None:
    """The structured parse, which is the point of the deterministic parser."""
    parse = result.parse
    st.markdown("**Query interpretation**")
    rows = {
        "Entity": parse.entity_label or "—",
        "Relation": ", ".join(str(r.value) for r in parse.relations) or "—",
        "Direction": parse.direction.value,
        "Intent": parse.intent.value,
        "Matched as": parse.entity_match or "—",
    }
    st.dataframe(
        [{"Field": k, "Value": v} for k, v in rows.items()],
        hide_index=True,
        width="stretch",
    )
    if parse.ambiguous:
        st.info(
            "That phrase maps to more than one relation in the specification's synonym "
            "table, so every matching relation was searched rather than guessing one.",
            icon=":material/help:",
        )


def result_cards(result) -> None:
    if result.paths:
        st.markdown(f"**{len(result.paths)} path(s) found**")
        for path in result.paths:
            rows = " &nbsp;".join(
                [
                    f"`{path[0]['from']}`",
                    *[
                        f"──&nbsp;`{step['relation']}`&nbsp;──▶&nbsp;`{step['to']}`"
                        for step in path
                    ],
                ]
            )
            st.markdown(rows)
            st.caption(
                " → ".join(
                    f"{step['from']} --{'→' if step['direction'] == 'forward' else '←'}--> "
                    f"{step['relation']} --> {step['to']}"
                    for step in path
                )
                + "  (a ← hop means the stored edge points the other way)"
            )
        return

    if not result.answers:
        st.warning(result.message or "No answer found.", icon=":material/search_off:")
        return

    st.markdown(f"**{len(result.answers)} result(s) found**")
    for answer, (subject, relation, obj) in zip(
        result.answers, result.matched_triples, strict=True
    ):
        # The engine orients each triple query-entity-first, which for most of these
        # questions puts the answer on the right. Render the stored direction with an
        # arrow so "incoming" reads correctly instead of needing a "(reverse)" label.
        if subject == answer:
            line = f"{answer} &nbsp;──&nbsp;{relation}&nbsp;──▶&nbsp; {obj}"
        else:
            line = f"{subject} &nbsp;──&nbsp;{relation}&nbsp;──▶&nbsp; {answer}"
        with st.container(border=True):
            st.markdown(
                f"<span class='kg-answer'>{answer}</span>"
                f"&nbsp;&nbsp;<span class='kg-muted'>{line}</span>",
                unsafe_allow_html=True,
            )


def evidence_panel(result) -> None:
    if not result.evidence:
        return
    st.markdown("**Supporting evidence**")
    for item in result.evidence:
        with st.container(border=True):
            source = item.get("document_id", "?")
            st.caption(f"Source: {source}.txt")
            st.markdown(
                f"<div class='kg-evidence-quote'>“{item.get('sentence', '')}”</div>",
                unsafe_allow_html=True,
            )
            resolved = _resolution_line(item)
            if resolved:
                st.markdown(resolved, unsafe_allow_html=True)
            st.caption(
                f"Confidence {float(item.get('confidence', 0)):.2f} — a rule-confidence "
                "label, not a statistical probability."
                + ("  ·  Inferred, not stated literally." if item.get("inferred") else "")
            )


def _resolution_line(item: dict) -> str:
    """The 'Resolved: "He" → Arun Mehta' line that shows coreference was needed.

    Empty when the text already named both endpoints, because then coreference added
    nothing and claiming otherwise would oversell the pipeline.
    """
    pairs = [
        f'“{r["from"]}” &rarr; <b>{r["to"]}</b>'
        for r in item.get("resolutions", [])
        if r.get("from") and r.get("to")
    ]
    if not pairs:
        return ""
    return "<span class='kg-resolved'>Resolved: " + " &nbsp;·&nbsp; ".join(pairs) + "</span>"


def triple_table(kb: KnowledgeBase) -> None:
    st.markdown("**Extracted triples**")
    rows = [
        {
            "Subject": t.subject,
            "Relation": t.relation,
            "Object": t.object,
            "Confidence": f"{t.confidence:.2f}",
            "Source": f"{t.document_id}.txt",
            "Inferred": t.inferred,
        }
        for t in kb.triples
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def pipeline_details(kb: KnowledgeBase) -> None:
    """Debug mode: exactly how the graph was built (specification section 24)."""
    docs_dir = AppConfig.from_env().docs_dir
    for document_id in sorted({t.document_id for t in kb.triples} | {str(d) for d in kb.report.get("counts", {}).get("document_ids", [])}):
        path = docs_dir / f"{document_id}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""

        with st.expander(document_id):
            st.markdown("**Document**")
            st.text(text or "(not found)")

            doc_triples = [t for t in kb.triples if t.document_id == document_id]
            if doc_triples:
                sentences = list(
                    dict.fromkeys(
                        (t.sentence_index, t.sentence) for t in doc_triples
                    )
                )
                st.markdown("**Sentences**")
                st.dataframe(
                    [{"#": i, "Text": s} for i, s in sentences],
                    hide_index=True,
                    width="stretch",
                )

            st.markdown("**Entities and coreference**")
            rows = [
                {
                    "Mention": r["mention"],
                    "Resolved entity": r["entity"] or "— unresolved —",
                    "Type": r["mention_source"],
                    "Source": r["source"],
                    "Anaphoric": r["is_anaphoric"],
                }
                for r in kb.resolutions
                if r["document_id"] == document_id
            ]
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.caption("Run a rebuild to capture per-mention resolution detail.")

            st.markdown("**Relations**")
            st.dataframe(
                [
                    {
                        "Subject": t.original_subject,
                        "Relation": t.relation,
                        "Object": t.original_object,
                        "Confidence": f"{t.confidence:.2f}",
                        "Evidence": t.sentence,
                    }
                    for t in doc_triples
                ]
                or [{"Subject": "—"}],
                hide_index=True,
                width="stretch",
            )
            if doc_triples:
                st.markdown("**Triples**")
                st.code(
                    "\n".join(
                        f"({t.subject}, {t.relation}, {t.object})" for t in doc_triples
                    ),
                    language="text",
                )


# ------------------------------------------------------------------------------- page

st.html(PAGE_CSS)

config = AppConfig.from_env()

# ------------------------------------------------------------------- demo mode startup
# Spec section 33: launch in a useful state rather than an empty shell.
if not artifacts_present(config.artifacts_dir):
    with st.spinner("No artifacts found — building the knowledge graph on first run…"):
        rebuild(token=0)
    st.success("Knowledge graph built. Artifacts written to `artifacts/`.")

kb = load_knowledge_base(config.artifacts_dir, st.session_state.get("rebuild_token", 0))

st.markdown("<div class='kg-title'>Knowledge graph explorer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='kg-subtitle'>NLP + coreference + relation extraction MVP</div>",
    unsafe_allow_html=True,
)

kpi_row(kb)

# -------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    entity_types = st.multiselect(
        "Entity type",
        kb.entity_types,
        default=kb.entity_types,
        help="Filter the graph by entity type.",
    )
    relations = st.multiselect(
        "Relation type",
        kb.relation_types,
        default=kb.relation_types,
        help="Filter the graph by relation type.",
    )
    st.caption(swatch_legend(kb.entity_types), unsafe_allow_html=True)

    if st.button("Show full graph", width="stretch", icon=":material/refresh:"):
        for key in ("entity_types", "relations"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    search = st.text_input("Search entity", placeholder="Nila Rao")
    focus_id = None
    if search:
        needle = search.strip().lower()
        for node_id, attrs in kb.graph.nodes(data=True):
            if needle in {str(attrs.get("label", "")).lower(), node_id.lower()} or any(
                needle in str(a).lower() for a in attrs.get("aliases", [])
            ):
                focus_id = node_id
                break

    st.divider()
    show_details = st.checkbox("Show NLP pipeline details", value=False)

    st.divider()
    if st.button("Rebuild knowledge graph", width="stretch", type="primary", icon=":material/build:"):
        st.session_state["rebuild_token"] = st.session_state.get("rebuild_token", 0) + 1
        with st.status("Rebuilding…", expanded=True) as status:
            outcome = rebuild(token=st.session_state["rebuild_token"])
            for message in outcome["messages"]:
                st.write(message)
            status.update(label="Done.", state="complete")
        st.cache_data.clear()

# --------------------------------------------------------------------------------- graph
visible = filter_graph(kb.graph, entity_types, relations)

with st.container(border=True):
    st.markdown("**Knowledge graph**")
    if visible.number_of_nodes() == 0:
        st.warning("No entities match the current filters.", icon=":material/filter_alt_off:")
    else:
        st.caption(
            f"{visible.number_of_nodes()} nodes · {visible.number_of_edges()} edges · "
            "drag, zoom, hover a node or edge"
        )
        st.iframe(
            render_graph(visible, focus=focus_id),
            width="stretch",
            height=660,
        )

if focus_id:
    attrs = kb.graph.nodes[focus_id]
    st.markdown(f"**{attrs.get('label')}** &nbsp;·&nbsp; {attrs.get('type')}", unsafe_allow_html=True)
    links = connections_of(kb.graph, focus_id)
    if links:
        st.dataframe(
            [
                {
                    "Direction": row["direction"],
                    "Relation": row["relation"],
                    "Other": row["other"],
                    "Confidence": f"{float(row.get('confidence') or 0):.2f}",
                }
                for row in links
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No connections.")

# ---------------------------------------------------------------------------------- query
query_col, interpretation_col = st.columns([1, 1], gap="medium")

with query_col, st.container(border=True, height=280):
    st.markdown("**Query engine**")
    typed = st.text_input(
        "Query",
        placeholder="Who founded Aether Analytics?",
        label_visibility="collapsed",
    )
    chosen = st.selectbox(
        "Example queries",
        ["— pick an example —", *EXAMPLE_QUERIES],
        label_visibility="collapsed",
    )
    question = typed or (chosen if chosen != "— pick an example —" else "")
    execute = st.button("Execute", icon=":material/search:", type="primary")

with interpretation_col, st.container(border=True, height=280):
    result = None
    if question:
        engine = QueryEngine(kb.graph, get_registry())
        result = engine.run(question)
        interpretation_panel(result)
    else:
        st.markdown("**Query interpretation**")
        st.caption("Run a query to see how it was parsed.")

if result is not None:
    st.markdown("**Query results**")
    result_cards(result)
    evidence_panel(result)

st.divider()
triple_table(kb)

if show_details:
    st.divider()
    st.markdown("**NLP pipeline details**")
    pipeline_details(kb)

with st.sidebar:
    st.divider()
    st.caption("Graph derived from five documents. No LLM, no external API, no network.")
