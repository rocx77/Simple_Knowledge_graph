# Engineering Guidelines

> **Status:** active · **Last updated:** Phase 3
> Derived from the charter. These are the rules that make the system debuggable.
> When a rule and convenience conflict, the rule wins.

## 1. Layering and dependency direction

```
app.py                (Streamlit entry point; UI composition only)
   |
ui/                   (theme, reusable components, views)
   |
src/                  (pipeline + query + visualisation -- pure Python)
   |
data/, artifacts/     (inputs and outputs)
```

**Rule L1 — the pipeline must never import the UI.**
No module under `src/` may import `streamlit`, `pyvis`, or `app`. This is why the
pipeline is testable and scriptable: `python -m src.pipeline` is a first-class entry
point. Verified by `tests/test_architecture.py`.

**Rule L2 — dependencies point downward only.**
`ui/` may import `src/`. `src/` must not import `ui/`. `scripts/` and `tests/` may
import anything. No circular imports between sibling modules.

**Rule L3 — one responsibility per module.**
If a module's name contains "and", it probably has two responsibilities and should be
split. `config.py` holds settings only. `models.py` holds dataclasses only.

## 2. Object-oriented design principles

**Rule O1 — classes own behaviour, dataclasses own data.**
Plain `@dataclass` records describe *what something is*. Behaviour lives in a class
that takes them as input. Example: `Triple` is a dataclass; `TripleBuilder` is the
class that decides which triples are legal.

**Rule O2 — dependency injection via constructor, never via module globals.**
`EntityResolver(config, registry)` not `EntityResolver()` reaching for a global.
This is what makes rules individually testable without running the pipeline.

**Rule O3 — extend, don't modify.**
New extraction logic is a new `RelationRule` subclass registered in the rule list.
Existing rules are never edited to accommodate a new sentence shape. If a rule needs
a generalisation, add a parameter or a sibling rule.

**Rule O4 — immutable value objects by default.**
Frozen dataclasses (`@dataclass(frozen=True)`) for records that are produced and then
only read: `Triple`, `CanonicalEntity`, `Query`. Mutable only where the pipeline
genuinely accumulates (`PipelineResult`, the graph).

**Rule O5 — explicit enums for closed vocabularies.**
Entity type, relation name, resolution source, intent and direction are `Enum`
subclasses with a `str` base so they serialise to readable JSON. Never bare strings
for a closed set.

## 3. Typed data over dictionaries

**Rule T1 — no untyped dicts in the pipeline's internal contracts.**
Use dataclasses with type hints. Dicts are acceptable only at the serialisation
boundary (JSON/CSV) and in the UI's transient display state.

**Rule T2 — every dataclass documents its invariants** in its own docstring. If an
invariant is not obvious from the field list, it is documented there.

## 4. Naming

**Rule N1 — names describe the domain, not the implementation.**
`RelationRule`, not `RuleV2` or `Handler5`. `docs/02_PIPELINE_PLAN.md` and code must
use the same vocabulary.

**Rule N2 — `rule_id` is a stable public identifier.** `R_FOUNDED_PASSIVE`. It appears
in artifacts, in tests, and in the UI's explainability trace. Renaming one is a
breaking change to the report.

**Rule N3 — canonical relations use lower snake case without the `R_` prefix.**
`works_on`, not `R_WORK_ON`. The `R_` prefix belongs to the rule id, not the relation.

**Rule N4 — no abbreviations** except `doc`, `id`, `nlp`, `pos`, `dep`, `kg`. Spell out
`coreference`, `resolution`, `dependency`.

## 5. Error handling

**Rule E1 — never let one failure kill the run.**
Every stage catches per-item errors and records them. A sentence that fails relation
extraction must not stop the other 19 sentences. Specification section 36.

**Rule E2 — degrade, then report.**
The failure path is: catch → log a warning with context → record a diagnostic entry →
continue. Diagnostics surface in `pipeline_report.json` and the UI's debug view, so
degradation is visible rather than silent.

**Rule E3 — a typed exception hierarchy, not bare `except Exception` at call sites.**
`KnowledgeGraphError` is the root. Subclasses: `ConfigurationError`,
`ModelUnavailableError`, `CorpusError`, `QueryError`. Bare `except` is only acceptable
in the top-level entry point and in `scripts/verify_*.py`, where the job is to
report rather than to recover.

**Rule E4 — user-facing messages are part of the product.**
`No matching entity found for "XYZ".` is a defined string in the query engine, tested
as such. Not an ad-hoc f-string at the raise site.

**Rule E5 — fail fast on misconfiguration, fail soft on bad data.**
An unset required path or an unreadable corpus is a configuration error and should
stop the run. An individual malformed sentence is data and should not.

## 6. Observability

**Rule O6 — every stage emits structured log records** tagged with the stage name and
document id where applicable.

**Rule O7 — model-loading noise is suppressed at the boundary.**
spaCy, torch, transformers and FastCoref all log at INFO on import and model load.
`src/logging_utils.py` silences those specific loggers so the pipeline's own output
stays readable. The pipeline's logger is never silenced.

**Rule O8 — timings are part of the report.** Each stage records elapsed seconds in
`pipeline_report.json`. Required for the performance claims in the runbook.

## 7. No hidden magic

**Rule M1 — no opaque shortcuts.** The specification forbids answering a known query
by string comparison. There is no `if query == "Who founded Aether Analytics?"`.

**Rule M2 — the graph is derived, never declared.** No module contains a literal list
of final triples. Tests assert the *output*, and the output must come from the text.

**Rule M3 — the entity registry is declared, and that is allowed.** A canonical entity
dictionary with aliases is explicitly permitted. It is a *lexicon*, not an answer key:
it says what names exist, not what relations hold between them.

**Rule M4 — every constant that encodes a linguistic judgement is named and located in
one place** (`src/rules/registry.py`, `src/query_parser.py`'s synonym map,
`src/entity_resolver.py`'s alias table). A reviewer must be able to find all
domain knowledge in the codebase without reading logic.

**Rule M5 — heuristics carry a label.** `confidence` is a heuristic label
(1.00 / 0.90 / 0.80), documented as such in code, in the artifacts, and in the UI. Never
presented as a probability.

## 8. Performance

**Rule P1 — expensive resources are loaded once and cached.**
spaCy `Language`, the FastCoref model, and the built graph are cached at their owner,
not per call. In the UI they are additionally wrapped in `st.cache_resource` /
`st.cache_data`.

**Rule P2 — rebuild is a distinct operation from query.**
Queries are graph traversals and must not trigger NLP. If a query re-runs the
pipeline, that is a bug.

**Rule P3 — no premature optimisation.** The corpus is sixteen documents. Optimise only
what a measurement shows is slow, and record the measurement.

## 9. Testing

**Rule S1 — every relation rule has its own test** asserting the triple it produces from
a named sentence. If a rule cannot be tested in isolation, it is doing too much.

**Rule S2 — tests assert behaviour, not implementation.** Assert triples, answers and
evidence — not internal call sequences.

**Rule S3 — expensive fixtures are session-scoped.** The spaCy model, the coref model
and the pipeline result are built once per test session.

**Rule S4 — slow tests are marked** `@pytest.mark.slow` so the fast path
(`pytest -m "not slow"`) stays usable during development.

**Rule S5 — an assertion failure must name the stage that produced the bad data.**
Include `document_id`, `sentence_id`, and `rule_id` in assertion messages. A failure
that says only "expected 16 triples" costs an hour; one that says "doc_03 s2 via
R_USES_ACTIVE" costs a minute.

## 10. Documentation discipline

**Rule D1 — `docs/02_PIPELINE_PLAN.md` is the stage contract.** If the plan and the
code disagree, one is a bug. The plan is updated in the same commit.

**Rule D2 — a new relation rule updates `docs/04_RELATION_RULES.md`** with its
trigger, dependency signature, output, confidence and rationale.

**Rule D3 — the corpus is immutable.** If `scripts/verify_corpus.py` fails, the text
was changed by mistake. Restore it; do not update the manifest to match.

**Rule D4 — an architectural fork gets an ADR.** When a decision has real, competing
alternatives and a non-obvious cost (not merely "I picked X"), record it in
`docs/adr/NNNN-slug.md` with context, the options considered, the decision, and the
consequences. Numbering is sequential and never reused. This applies even when the
decision looks forced in hindsight — the value is in the *rejected* options and the cost.

Current ADRs:

| # | Decision |
|---|---|
| 0001 | [Offline graph rendering](adr/0001-offline-graph-rendering.md) — PyVis cannot emit offline HTML; inline its bundled vis.js. |
| 0002 | [UI layering and layout](adr/0002-ui-layering.md) — `app.py` + `ui/`, AST-based import checks, caching boundaries. |

### Documentation index

| Document | Role |
|---|---|
| `00_PROJECT_CHARTER.md` | scope, invariants, non-goals |
| `02_PIPELINE_PLAN.md` | the stage contract (Rule D1) |
| `04_RELATION_RULES.md` | rule catalogue, confidence bands, declined readings (Rule D2, invariant I7/I8) |
| `05_QUERY_LANGUAGE.md` | query grammar, direction rule, synonym ambiguity, connection search |
| `06_ENGINEERING_GUIDELINES.md` | this file |
| `adr/` | architectural decision records (Rule D4) |
| `HANDOFF_DESKTOP_APP.md` (repo root) | desktop shell + screenshot harness, environment findings |

## 11. Debugging playbook

When a triple is wrong, work backwards along the chain — this order is the reverse of
the pipeline and is the fastest path to the defect:

| Observed | Likely stage | Check |
|---|---|---|
| Entity not detected at all | mention detection / EntityRuler | `scripts/verify_corpus.py`, ruler patterns, mention table |
| Entity detected, wrong type | EntityRuler missing or ordered wrong | ruler must be after `ner`; overwrite behaviour |
| Pronoun became a node | coreference | `MentionResolution.source` for that span |
| Pronoun resolved to wrong entity | coreference | antecedent order and gender agreement |
| Relation missing | rule match | dependency signature in the plan vs. actual parse |
| Relation present, wrong direction | rule | `RelationCandidate` inversion logic |
| Literal leaked in as a named entity | entity resolver | alias table overlap |
| Query returns nothing | query parser | parse result before traversal |

`Show NLP Pipeline Details` in the UI renders this chain for any document, which is
the intended way to debug during a live demo.