# Project Charter — Knowledge Graph MVP

> **Status:** active · **Last updated:** Phase 3 · **Owner:** NLP case study
> This document is the anchor. Any change to scope, architecture or expected output
> must be justified here first. If code and this charter disagree, one of them is a bug.

## 1. Objective

Demonstrate a complete, reproducible, fully-local **NLP → knowledge graph** pipeline that
converts a small corpus of English text into an interactive, queryable knowledge graph
where every answer is traceable back to the exact sentence that produced it.

The deliverable is a **case-study demonstration**, not a production system. It is judged
on: reliability on the provided corpus, explainability, modularity, visual polish, and
ease of live demonstration.

## 2. Pipeline shape (the "money slide")

```
DOCUMENTS
  -> spaCy (tokenise, POS, dependency parse, NER, noun chunks)
  -> FastCoref (neural coreference)  +  deterministic fallback layer
  -> EntityRuler + canonical entity registry (entity resolution)
  -> Rule-based relation extraction (deterministic, no LLM)
  -> Triple generation with full provenance
  -> NetworkX MultiDiGraph
  -> JSON/CSV serialisation
  -> PyVis interactive bubble graph  +  Streamlit UI
  -> Deterministic query parser -> graph traversal -> answer + evidence
```

## 3. Scope — in

| Capability | Detail |
|---|---|
| Corpus | The 5 fixed documents in `data/docs/`, verified byte-for-byte by `scripts/verify_corpus.py` |
| Entity inventory | 3 PERSON, 3 ORGANIZATION, 1 PRODUCT, 1 PROJECT, 2 COMPONENT |
| Relation inventory | 13 canonical relations, 2 optional (`contains`, `reuses`) |
| Coreference | FastCoref with a deterministic alias + antecedent fallback |
| Graph | In-memory `networkx.MultiDiGraph`, serialised to JSON + CSV |
| Query engine | Declarative template parser; `relation_lookup` + `connection_path` |
| UI | Streamlit + PyVis, offline-capable, demo-ready |
| Tests | Automated, mapping every required query and acceptance criterion |

## 4. Non-goals (explicitly out of scope)

These are **banned** by specification. Adding any of them invalidates the case study.

- No LLM / generative model of any kind, at any stage.
- No external API, cloud service, or network access **at runtime**.
- No Neo4j, Redis, Elasticsearch, Kafka, Docker, LangChain, or vector database.
- No semantic embeddings for query parsing. No fuzzy entity matching.
- No scaling work. The corpus is 30 documents; premature optimisation is a defect.
- No persistence beyond the four JSON/CSV artifacts.

## 5. Non-negotiable invariants

These hold for every implementation change. They are asserted by `tests/`.

| # | Invariant | Enforced by |
|---|---|---|
| I1 | The pipeline never imports `streamlit` or `pyvis`. `python -m src.pipeline` runs headless. | `tests/test_architecture.py` |
| I2 | No hard-coded query answers. No hard-coded graph. The graph is derived from the documents only. | `tests/test_acceptance.py` |
| I3 | A pronoun (`he`, `she`, `it`, `the bank`) never becomes a graph node. | `tests/test_coreference.py` |
| I4 | Every triple carries document, sentence, original mention and resolution source. | `tests/test_serialization.py` |
| I5 | One failing sentence never aborts the pipeline. Failures degrade to warnings. | `tests/test_pipeline.py` |
| I6 | Device is resolved at runtime (`cuda:0` if available, else `cpu`), overridable via `COREF_DEVICE`. Never GPU-only. | `src/config.py`, `scripts/verify_setup.py` |
| I7 | `confidence` is a deterministic rule-confidence **label**, never a statistical probability. Stated as such in the UI. | `docs/04_RELATION_RULES.md` |
| I8 | Inferred relations are flagged `inferred=True` and never presented as literally stated. | `tests/test_relations.py` |
| I9 | Dependencies stay on `transformers` 4.x. Relaxing this breaks FastCoref. | `requirements.txt`, `scripts/verify_setup.py` |
| I10 | Vis.js is embedded in the generated HTML. The UI works with no network. | `src/visualization.py` |

## 6. Expected output (the contract)

The pipeline **must** produce all 14 core triples from specification section 34:

```
(Dr. Mira Sen, founded, Aether Analytics)      (Arun Mehta, works_at, Aether Analytics)
(Aether Analytics, develops, OrionEdge)        (Arun Mehta, works_on, OrionEdge)
(Dr. Mira Sen, leads, Project Aurora)         (Arun Mehta, collaborates_with, Nila Rao)
(Project Aurora, focuses_on, graph-based      (Nila Rao, leads, Project Aurora)
                transaction analysis)          (Aether Analytics, partners_with, Quantum Forge)
(Nila Rao, improves, Entity Resolution        (OrionEdge, deployed_at, Helios Bank)
               Engine)                        (Helios Bank, uses, OrionEdge)
(Project Aurora, integrated_into, OrionEdge)  (Helios Bank, customer_of, Aether Analytics)
```

Plus two **optional** relations, enabled by default because they are linguistically
reliable and add demonstration value:

```
(Project Aurora, contains, Entity Resolution Engine)   <- "inside"
(Project Aurora, contains, Graph Matching Module)      <- possessive "Aurora's"
```

Target: **94 triples, 85 nodes, 94 edges.**

## 7. Required queries (the acceptance contract)

All 12 must return the exact expected entities. See `docs/05_QUERY_LANGUAGE.md`.

| Query | Expected answer | Relation | Direction |
|---|---|---|---|
| Who founded Aether Analytics? | Dr. Mira Sen | `founded` | incoming |
| What does Aether Analytics develop? | OrionEdge | `develops` | outgoing |
| Who works at Aether Analytics? | Arun Mehta | `works_at` | incoming |
| Who works on OrionEdge? | Arun Mehta | `works_on` | incoming |
| Who leads Project Aurora? | Dr. Mira Sen, Nila Rao | `leads` | incoming |
| Who collaborates with Arun Mehta? | Nila Rao | `collaborates_with` | both |
| Who partnered with Aether Analytics? | Quantum Forge | `partners_with` | both |
| Where is OrionEdge deployed? | Helios Bank | `deployed_at` | outgoing |
| Who uses OrionEdge? | Helios Bank | `uses` | incoming |
| Who improved the Entity Resolution Engine? | Nila Rao | `improves` | incoming |
| What is integrated into OrionEdge? | Project Aurora | `integrated_into` | incoming |
| Who is the customer of Aether Analytics? | Helios Bank | `customer_of` | incoming |

Bonus: `How is <A> connected to <B>?` — BFS path, max length 3.

## 8. Definition of done

1. `scripts/verify_setup.py` exits 0.
2. `scripts/verify_corpus.py` reports INTACT.
3. `python -m src.pipeline` regenerates all four artifacts.
4. `pytest` passes with zero failures, including all 12 queries and the acceptance map.
5. `streamlit run app.py` starts and renders the full graph.
6. Every answer in the UI shows its evidence chain.
7. `PROJECT_REPORT.md` is current.

## 9. Change protocol

Any change to this project must:

1. State which charter section it touches.
2. Update `docs/02_PIPELINE_PLAN.md` if it alters a stage contract.
3. Add or update a test **in the same commit** as the behaviour change.
4. Add an ADR to `docs/adr/` if it changes a technology or a structural decision.
5. Update `PROJECT_REPORT.md`.

A change that violates section 4 (non-goals) requires explicit sign-off and a new ADR.