# Pipeline Plan — Stage Contracts

> **Status:** active · **Last updated:** Phase 3 (all 8 stages specified)
> This is the implementation contract. Code must match it; if code must differ, this
> document changes in the same commit. Governing rules: `docs/00_PROJECT_CHARTER.md`
> (scope) and `docs/06_ENGINEERING_GUIDELINES.md` (how).

Dependency signatures in section 6 were verified empirically against
spaCy 3.8.16 / `en_core_web_sm`, not assumed. Where the parse is unusual the plan
says so, because that is where bugs live.

---

## S0 — Foundations

**Purpose** Config, errors, logging, typed models, canonical id scheme.

| Item | Design |
|---|---|
| `src/config.py` | Frozen `AppConfig` dataclass. Fields: `docs_dir`, `artifacts_dir`, `spacy_model`, `coref_model`, `coref_device`, `enable_optional_relations`, `log_level`, `max_path_length`. Loaded by `AppConfig.from_env()`. Env vars: `KG_DOCS_DIR`, `KG_ARTIFACTS_DIR`, `KG_SPACY_MODEL`, `COREF_MODEL_NAME_OR_PATH`, `COREF_DEVICE`, `KG_ENABLE_OPTIONAL_RELATIONS`, `KG_LOG_LEVEL`. |
| Device resolution | `COREF_DEVICE` if set and usable → else `cuda:0` if `torch.cuda.is_available()` → else `cpu`. An unusable `cuda:*` request with no CUDA falls back to `cpu` with a warning. Never raises. Satisfies invariant I6. |
| `src/errors.py` | `KnowledgeGraphError` root → `ConfigurationError`, `ModelUnavailableError`, `CorpusError`, `RelationExtractionError`, `QueryError`. |
| `src/logging_utils.py` | `configure_logging(level)` silences `httpx`, `huggingface_hub`, `urllib3`, `transformers`, `datasets`, `fastcoref`, `sentence_transformers`, `absl`, `filelock`, `numba` to WARNING+. Leaves our own loggers intact (rule O7). |
| `src/ids.py` | `make_entity_id(type, label) -> "person:arun_mehta"`, `make_triple_id(subject_id, relation, object_id, document_id)`. Normalises to lower snake case; strips honorifics (`Dr.` → `mira_sen`); strips possessive `'s`. |
| `src/models.py` | Frozen dataclasses: `Document`, `EntityMention`, `CanonicalEntity`, `RelationCandidate`, `Triple`, `MentionResolution`, `TokenView`, `SentenceView`, `Query`, `QueryResult`, `Evidence`, `TraceStep`, `StageReport`, `PipelineResult`. Enums: `EntityType`, `RelationName`, `ResolutionSource`, `QueryIntent`, `QueryDirection`, `MentionSource`, `ConfidenceBand`. |

**Failure modes** Missing model → `ModelUnavailableError` with the exact install
command in the message. Missing corpus dir → `CorpusError`. Bad env value → warn and
default.

**Acceptance** `src.config.resolve_device()` returns `cpu` here and `cuda:0` when
CUDA is present; `tests/test_config.py` covers the matrix.

---

## S1 — Document ingestion

**Purpose** Load the corpus into `Document` records.

- Sorted glob `*.txt` under `data/docs/`. `document_id` = filename stem
  (`doc_01_company`).
- **Never concatenate.** Documents are processed independently end to end, because
  FastCoref is per-document and cross-document coreference is out of scope.
- Read UTF-8 with `errors="replace"`; record a warning if replacement occurred.
- **Skip and log** zero-length and whitespace-only files (rule E2), rather than
  feeding them to spaCy.
- Records: `character_count`, `sentence_count` (post spaCy), `sha256`.

**Failure modes** Unreadable file → warning, file skipped, run continues. Empty corpus →
`CorpusError` (this is configuration, not data — rule E5).

**Acceptance** 24 documents loaded, hashes match `data/corpus_manifest.json`.

---

## S2 — spaCy processing

**Purpose** Linguistic backbone: tokens, POS, dependencies, NER, noun chunks.

- One shared `spacy.load(config.spacy_model)`. Loaded once, cached (rule P1).
- **EntityRuler added after `ner`** so it overwrites spaCy's incorrect labels. This is
  not optional — see S6 note.
- Ruler patterns are **named aliases only**: `Aether Analytics`, `Arun Mehta`,
  `Nila Rao`, `Dr. Mira Sen`, `Mira Sen`, `Quantum Forge`, `Helios Bank`, `OrionEdge`,
  `Project Aurora`, `Aurora`, `Entity Resolution Engine`, `Graph Matching Module`.
  Longest patterns first so `Project Aurora` wins over `Aurora`, and
  `Dr. Mira Sen` wins over `Mira Sen`.
- **Definite descriptions and pronouns are deliberately NOT ruler patterns.**
  `the bank`, `she`, `the platform` are handled by coreference. Putting them in the
  ruler would create entities with no canonical identity and corrupt NER.
- Retained per token (spec section 6): `text`, `pos_`, `dep_`, `head.text`,
  `ent_type_`, `start_char`, `end_char`, plus `lemma_`, `morph`, `i`.
- Retained per sentence: index, text, `start_char`, `end_char`.
- Retained noun chunks: text and char span.

**Mention detection.** spaCy NER will not surface `The company`, `The bank`,
`the platform`, `the engineer`, `the project`. A dedicated `MentionDetector`
produces candidate mentions from three sources, tagged by `MentionSource`:

1. `RULER` — ruler spans (highest trust).
2. `SPACY_NER` — NER spans not overridden by the ruler.
3. `COMMON_NOUN` — a noun chunk whose head noun is in the definite-description table
   (`company`, `bank`, `platform`, `project`, `engineer`, `extension`, `engine`), or a
   possessive pronoun (`its`, `their`).

`COMMON_NOUN` is the mechanism that lets `The company develops OrionEdge` become
`(Aether Analytics, develops, OrionEdge)`.

**Acceptance** Every expected mention found with correct type; the four
`COMMON_NOUN` cases in the corpus are detected.

---

## S3 — Coreference resolution

**Purpose** Map anaphoric mentions to canonical entities, without inventing nodes.

Two collaborating components:

**`FastCorefResolver`** — wraps `fastcoref.FCoref`, device from config, **reuses the
pipeline's own spaCy `Language` object** (FastCoref needs one for candidate spans;
loading a second copy would double memory and double load time). Char spans are
converted to `MentionResolution` with `source=FASTCOREF`.

**`DeterministicCoreferenceFallback`** — the specification's precedence chain:

1. `FASTCOREF` — neural cluster.
2. `ALIAS` — exact match in the canonical alias table (`the company` → Aether
   Analytics, `the bank` → Helios Bank, `the platform` → OrionEdge, `the project` →
   Project Aurora, `the engineer` → Arun Mehta, `she` → Nila Rao, `he` → Arun Mehta).
3. `ANTECEDENT` — nearest preceding mention in the same document that is type
   compatible, scored by: POS distance (prefer same sentence), type compatibility, and
   **gender agreement from `token.morph`** (`He` → `Gender=Masc` → a PERSON;
   `She` → `Gender=Fem`). Organisation antecedents are **never** filtered by number
   agreement, because spaCy mislabels `Aether Analytics` as `Number=Plur` in doc_03.
4. `UNRESOLVED` — recorded as a diagnostic and **never** promoted to a node (I3).

The resolver returns **both** the original mention and the resolved entity for every
resolution, with a `CONFLICT` flag when `FASTCOREF` and `ALIAS` disagree — conflicts
are recorded, not silently dropped.

**Why coreference runs before relation normalisation** — it is the whole point of the
demo. `He works on OrionEdge` is only `(Arun Mehta, works_on, OrionEdge)` because
S3 ran first, and the UI shows the original mention alongside the resolution.

**Failure modes** Model unavailable → warn, continue with fallback only, mark the run
degraded in `PipelineReport`. Unresolvable mention → diagnostic, no node.

**Acceptance** All 7 required mappings resolve, verified on the final normalised
output rather than on raw model clusters (spec section 35 explicitly permits this).

---

## S4 — Entity resolution and the canonical registry

**Purpose** One node per real-world entity.

- `data/entities.yaml` declares the canonical inventory: `id`, `label`, `type`,
  `aliases`. This is a **lexicon**, permitted by spec section 37 — it declares names,
  never relations.
- `EntityRegistry` loads and validates it, and builds the alias lookup with
  **longest-match-first** precedence so `Project Aurora` beats `Aurora` and
  `Entity Resolution Engine` beats `Engine`-style partials.
- Lookup order (spec section 20): exact canonical name → exact alias → normalised
  lowercase → possessive-stripped (`Aurora's` → `Aurora`).
- `EntityResolver` maps a mention span to a canonical `CanonicalEntity`, or to a
  `LiteralEntity` for a non-canonical noun phrase.
- `MentionAnnotator` writes the resolution onto each `EntityMention` and builds the
  per-document mention table for the UI.

**Literal nodes** `focuses_on` requires one (spec section 11). A `LITERAL` node is a
node whose `type` is `LITERAL` and whose `canonical` is `False`. These are the only
non-entity nodes permitted, and only for prepositional objects.

**Acceptance** 11 canonical entities resolvable; no two nodes for `OrionEdge` and
`the platform`.

---

## S5 — Relation extraction

**Purpose** Deterministic, dependency-driven extraction into the canonical relation
vocabulary. **No LLM.** No LLM. No LLM.

`RelationExtractor` iterates a registered list of `RelationRule` objects over every
sentence of every document. Each rule implements:

```python
class RelationRule(ABC):
    rule_id: str
    relation: RelationName
    optional: bool = False
    def matches(self, ctx: SentenceContext) -> bool: ...
    def extract(self, ctx: SentenceContext) -> list[RelationCandidate]: ...
```

`SentenceContext` bundles the sentence, its document, the mention annotations, and
resolved-entity access — rules receive everything they need so they stay small.

**The complete rule set.** Each row is verified against the real parse.

| `rule_id` | Trigger | Verified dependency signature | Emits | Conf. | Inferred |
|---|---|---|---|---|---|
| `R_FOUNDED_PASSIVE` | lemma `found`, passive | `subj[nsubjpass]` → verb(ROOT) → `by[agent]` → `agent[pobj]` | `(agent) founded (subj)` | 1.00 | no |
| `R_DEVELOPS_ACTIVE` | lemma `develop`/`build`/`create`, active | `subj[nsubj]` → verb(ROOT) → `obj[dobj]` | `(subj) develops (obj)` | 1.00 | no |
| `R_LEADS_ACTIVE` | lemma `lead`/`head`, active | `subj[nsubj]` → verb(ROOT) → `obj[dobj]` | `(subj) leads (obj)` | 1.00 | no |
| `R_FOCUSES_ON` | lemma `focus`, prep `on`/`upon` | `subj[nsubj]` → verb(ROOT) → `prep[prep]` → `obj[pobj]` | `(subj) focuses_on (literal)` | 1.00 | no |
| `R_JOINED_EMPLOYMENT` | lemma `join`/`hire` | `subj[nsubj]` → verb(ROOT) → `org[dobj]` | `(subj) works_at (org)` | 0.90 | no |
| `R_WORK_ON` | lemma `work`, prep `on` | `subj[nsubj]` → verb(ROOT) → `prep[prep]` → `obj[pobj]` | `(subj) works_on (obj)` | 0.90 | no |
| `R_COLLABORATE` | lemma `collaborate`, prep `with` | `subj[nsubj]` → verb(ROOT) → `prep[prep]` → `obj[pobj]` | `(subj) collaborates_with (obj)` | 0.90 | no |
| `R_PARTNER_WITH` | lemma `partner`, prep `with` | entity[ROOT] ← `verb[acl]` ← `prep[prep]` → `obj[pobj]` | `(subj) partners_with (obj)` | 1.00 | no |
| `R_DEPLOYED_AT` | lemma `deploy`/`install`/`run`, passive, prep `at`/`in` | `subj[nsubjpass]` → verb(ROOT) → `prep[prep]` → `obj[pobj]` | `(subj) deployed_at (obj)` | 1.00 | no |
| `R_USES_ACTIVE` | lemma `use`/`utilise`, active | `subj[nsubj]` → verb(ROOT) → `obj[dobj]` | `(subj) uses (obj)` | 0.90 | no |
| `R_IMPROVES` | lemma `improve`/`enhance`, active | `subj[nsubj]` → verb(ROOT) → `obj[dobj]` | `(subj) improves (obj)` | 1.00 | no |
| `R_INTEGRATED_INTO` | lemma `integrate`/`embed`, `into`/`in` | matrix `subj[nsubj]` → modal(ROOT) → `verb[xcomp]` → `thing[dobj]` + `prep[prep]` → `target[pobj]` | `(thing) integrated_into (target)`, context = matrix subject | 0.80 | **yes** |
| `R_CUSTOMER_OF` | lemma `renew`/`sign`, dobj `contract`, `with` | `subj[nsubj]` → verb(ROOT) → `contract[dobj]` → `prep[prep]` → `org[pobj]` | `(subj) customer_of (org)` | 0.80 | **yes** |
| `R_LEADS_IN_APPOS` | lemma `lead` inside an `appos` | `host[pobj]` ← `entity[appos]` ← `rel[amod, lemma=lead]` → `project[dobj]` | `(host) leads (project)` | 1.00 | no |
| `R_CONTAINS_INSIDE` | prep `inside`/`within`/`in` under a relation verb | `prep[prep]` → `container[pobj]` | `(container) contains (obj)` | 0.90 | no | **optional** |
| `R_CONTAINS_POSSESSIVE` | entity with `poss` head | `owner[poss]` → `part[dobj/compound-root]` | `(owner) contains (part)` | 0.90 | no | **optional** |

**Cross-cutting extractor rules, each forced by an observed parse quirk:**

- **E-R1 Prepositions attach to objects, not only to verbs.** In doc_05 `with` attaches
  to `contract`, in doc_02 `for` attaches to `work`. Prep search walks the whole verb
  subtree, never just direct children.
- **E-R2 Predicates are not always the root.** In doc_03 `partnered` is an `acl`
  modifier and `Aether Analytics` is the ROOT. Rules accept `ROOT`, `acl` and `acl:rel`
  headed predicates.
- **E-R3 Locate relation verbs by lemma scan within the sentence, not by dependency
  label alone.** In doc_02 the reduced relative `leading` is `amod`, not `acl`.
- **E-R4 Descend `xcomp` under a modal.** `plans to integrate` holds the real predicate.
- **E-R5 Expand objects via noun chunks, not the `pobj` head token.** `pobj=analysis`
  must become the chunk `graph-based transaction analysis`.
- **E-R6 Never cross a reporting verb.** `REPORTING_VERBS = {say, report, state, claim,
  note, mention, announce, argue, assert, believe, think, explain, write, tell}`.
  Without this, `Arun Mehta said [the extension will reuse Aurora's module]` fabricates
  `(Arun Mehta, reuses, Graph Matching Module)` — attributing Arun's opinion as his
  action. This is the single most important guard in the extractor.
- **E-R7 Candidates must survive endpoint resolution.** Both endpoints must resolve to
  known entities, otherwise the candidate becomes a *skipped* diagnostic with a reason,
  never a guessed endpoint.

**Deliberate non-extractions** (specification is silent; inventing facts is forbidden):

| Sentence | Why no triple |
|---|---|
| `She leads the entity resolution work for the project.` | `leads` object is a common noun, not an entity. Recorded as skipped. |
| `She reported that the engine reduced duplicate alerts by 18 percent.` | reporting verb (E-R6) |
| `...to provide GPU infrastructure for OrionEdge.` | purpose clause, no canonical relation |
| `The company will extend the platform to additional branches.` | `extend` is not in the vocabulary |
| `Arun Mehta said the extension will reuse Aurora's graph matching module.` | reporting verb **and** unresolvable subject `the extension` |

Each is expected to appear in `skipped_candidates` with a reason. Surfacing them in
the debug view is better for the case study than hiding them.

**Acceptance** All 14 core triples plus 2 optional are produced, each attributed to a
`rule_id`.

---

## S6 — Triple generation

**Purpose** Turn validated candidates into provenance-complete triples.

`TripleBuilder.build(candidates)`:
1. Drop candidates with an unresolved endpoint.
2. Resolve each endpoint to a canonical entity, recording the **original** surface
   form alongside the canonical form.
3. Apply the confidence band: **1.00** direct named-entity relation · **0.90** relation
   involving a successful coreference or a deterministic semantic normalisation ·
   **0.80** deterministic semantic inference. These are heuristic labels, not
   probabilities (I7).
4. Set `inferred` from the candidate, never inferred from confidence.
5. Set `coref_resolved` if any endpoint came from a non-canonical mention.
6. De-duplicate on `(subject_id, relation, object_id, document_id)`, keeping the
   highest-confidence instance and recording the count.

Every triple carries: `triple_id`, `subject`, `subject_id`, `relation`, `object`,
`object_id`, `document_id`, `sentence_id`, `sentence`, `original_subject`,
`original_object`, `coref_resolved`, `inferred`, `confidence`, `rule_id`.

**Acceptance** 94 triples; every field populated; the 92 required present.

---

## S7 — Graph construction and serialisation

**Purpose** `nx.MultiDiGraph`, then the four artifacts.

- One node per canonical entity **that participates in at least one triple**. A
  recognised-but-unconnected entity still appears in `entities.json` with a
  `has_edges: false` flag, so nothing is silently lost.
- Node attributes: `label`, `type`, `aliases`, `mention_count`, `documents`,
  `in_degree`, `out_degree`, `degree`.
- Edge attributes: `relation`, `confidence`, `document_id`, `sentence_id`, `sentence`,
  `inferred`, `rule_id`, `original_subject`, `original_object`, `triple_id`.
- `MultiDiGraph` so two relations between the same pair stay separate and provenance
  is never collapsed (spec section 13).
- Symmetry for querying is **not** stored — `collaborates_with` and `partners_with` are
  stored once, in their canonical direction, and the query engine searches both.

Artifacts, all human-readable JSON, indent 2:

| File | Contents |
|---|---|
| `artifacts/graph.json` | Node-link graph, vis.js compatible |
| `artifacts/triples.csv` | One row per triple, all provenance columns |
| `artifacts/entities.json` | Canonical registry + mention inventory + alias index |
| `artifacts/pipeline_report.json` | Stage timings, counts, device, library versions, warnings, unresolved mentions, skipped candidates, relation histogram |

**Acceptance** 85 nodes, 94 edges; all four files written; JSON round-trips.

---

## S8 — Pipeline orchestration

**Purpose** One object that runs the whole thing, with progress reporting.

`KnowledgeGraphPipeline.run(progress_cb=None)` executes S1→S7 in order, wrapping each
in a `StageReport` (name, elapsed, ok, warnings, error). A `progress_cb(message)`
callback drives the Streamlit progress messages required by spec section 32:
`Loading documents… / Running spaCy… / Resolving coreferences… / Extracting
relations… / Building graph… / Saving artifacts… / Done.`

**Acceptance** `python -m src.pipeline` regenerates all four artifacts and prints a
summary. No Streamlit import anywhere in `src/`.

---

## Stage dependency summary

```
S0 foundations
  └── S1 ingestion
        └── S2 spaCy            (shares one Language object)
              └── S3 coref      (reuses the S2 Language object)
                    └── S4 entity resolution
                          └── S5 relation extraction
                                └── S6 triples
                                      └── S7 graph + serialisation
                                            └── S8 orchestration
```

Stage 5 is the highest-risk stage. Its rule table in section S5 was derived from a
measured parse of the actual corpus, not from the grammar as assumed. If a rule fails
during implementation, check the actual parse first — the quirk is probably real.