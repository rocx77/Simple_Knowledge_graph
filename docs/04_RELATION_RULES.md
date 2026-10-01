# 04 — Relation rules

Authoritative description of how `(subject, relation, object)` triples are extracted.
This document is the enforcement point for charter invariant **I7** (confidence is a rule
label, never a probability) and **I8** (inferred relations are never presented as stated).

Implementation: `src/rules/base.py`, `src/rules/extraction.py`, `src/triple_builder.py`.
Acceptance tests: `tests/test_relation_extraction.py`.

---

## 1. Confidence is a label, not a probability

`ConfidenceBand` has exactly three values, and nothing in the pipeline produces anything
else:

| Band | Value | Meaning |
|---|---|---|
| `DIRECT` | `1.00` | Both endpoints named literally in the sentence. |
| `COREF_OR_NORMALISED` | `0.90` | At least one endpoint came from a coreference resolution or an alias/described-description normalisation. |
| `INFERRED` | `0.80` | The relation is not stated literally; it follows from a lexical trigger plus a licensed inference. |

**These are not probabilities.** They do not sum to 1, there is no confidence
distribution over them, and `0.90` does not mean "90% likely correct". They are an
ordinal statement about *how much interpretation the edge required* — which is the
question a reviewer of an extracted knowledge graph actually has.

The UI states this on every evidence card:

> Confidence 0.90 — a rule-confidence label, not a statistical probability.

Observed distribution over the demonstration corpus: 10 × `1.00`, 5 × `0.90`, 1 × `0.80`.

## 2. Inferred relations

An `inferred=True` triple is one whose relation is licensed by a rule rather than quoted.
The corpus produces exactly one: `Helios Bank customer_of Aether Analytics`, inferred from
the customer relationship in `doc_05_customer.txt`.

Inferred edges are marked in three places, because a reader who misses one of them would
take the edge at face value:

- `Triple.inferred` in `triples.csv`
- dashed and greyed edge styling in the graph, with `Inferred: yes — not stated literally`
  in the edge tooltip
- an `Inferred, not stated literally.` suffix on the evidence card

`GraphBuilder` never stores the inverse of an edge. Symmetry is a *query-time* decision,
not a stored fact, so an inferred `customer_of` does not silently imply that
Aether's customers include Helios as a stored edge.

## 3. Rule catalogue

Fifteen rules cover the thirteen spec-14 relations plus one that the corpus needs
(`contains`, from *"Project Aurora contains the graph matching module"*).
`works_on` has one rule and `contains` has two — postmodifying and possessive — so rules
and relations are not one-to-one.

### `reuses` has no rule

`reuses` is a member of the `RelationName` enum but **no extraction rule implements it**.
This is deliberate, and the reasoning is in `src/rules/base.py:47`:

`reuse` is the reason `REPORTING_VERBS` exists. In *"Arun Mehta said the extension will
reuse the graph matching module"*, the verb is quoted, so the speaker is **reporting** a
reuse rather than performing one. Without the reporting-verb guard the extractor emits
`(Arun Mehta, reuses, Graph Matching Module)` — attributing an opinion as an action
(parser fact P8).

Because the same sentence is the only licensing context for `reuses` in the corpus, and
because the correct reading of it is *not* a reuse by Arun, the relation is left
unimplemented rather than implemented incorrectly. The enum member is retained so the
lexicon is complete and so a future rule has a name to use. No output is lost: the
demonstration corpus contains no stated reuse relation.

If a reuse rule is ever added, it must sit behind `is_under_reporting_verb` and carry its
own test asserting that the quoted sentence produces nothing.

| Rule id | Relation | Trigger |
|---|---|---|
| `R_FOUNDED_PASSIVE` | `founded` | `X was founded by Y` (passive + agent) |
| `R_DEVELOPS` | `develops` | possessive or object-of-`develops` |
| `R_LEADS` | `leads` | `leads`, `leading` |
| `R_FOCUSES_ON` | `focuses_on` | `focuses on`, `focus on` |
| `R_JOINED_EMPLOYMENT` | `works_at` | `joined X as` / `joined X in` |
| `R_WORKS_ON` | `works_on` | `works on` |
| `R_COLLABORATES_WITH` | `collaborates_with` | `collaborates with` |
| `R_PARTNERS_WITH` | `partners_with` | `partnered with`, `partners with` |
| `R_DEPLOYED_AT` | `deployed_at` | `deployed at`, `deployed in` |
| `R_USES` | `uses` | `uses`, `utilizes` |
| `R_IMPROVES` | `improves` | `improved`, `improves`, `enhanced` |
| `R_INTEGRATED_INTO` | `integrated_into` | `integrated into`, `integrated in` |
| `R_CUSTOMER_OF` | `customer_of` | inferred from a customer relationship |
| `R_CONTAINS_INSIDE` | `contains` | `X contains Y` (postmodifying object) |
| `R_CONTAINS_POSSESSIVE` | `contains` | possessive form, `X's Y` |

## 4. Precision decisions that changed the output

These are the rules that matter, because each one was added to *remove* a wrong triple
rather than to add a right one. Reverting any of them reintroduces a known defect.

### 4.1 Reporting-clause suppression (`RuleContext.is_under_reporting_verb`)

`"The project focuses on graph-based transaction analysis."` contains a reporting verb
(`focuses`), which is exactly the trigger `R_FOCUSES_ON` looks for — but here the subject
is a project that does the focusing, and the parse is straightforward.

The suppression exists for the opposite case: in `"He leads the entity resolution work
for the project"`, the naive parse produces `(Nila Rao, leads, entity resolution work)` —
a plausible-looking triple about a common-noun object that nobody would defend. That
candidate is **rejected**, and the rejection is counted. The one skip in the corpus run
(`candidates=16 skipped=1`) is this candidate, and the one extra relation not expected.

### 4.2 Possessors resolve from their *own* span (`argument_for_own_token`)

`"The project focuses on graph-based transaction analysis."` The possessive-free path
yields the literal node, which is intended. But in `"OrionEdge is Helios Bank's primary
platform"` the relevant subject is the possessive noun inside the noun chunk, not the
chunk's head. Matching the chunk's head instead produced `(Helios Bank, deployed_at,
OrionEdge)`-shaped noise. `argument_for_own_token()` prefers the span that actually owns
the token when the token is a possessor.

### 4.3 Reduced relatives and appositives (`subject_tokens`)

`"The engineer collaborates with Nila Rao, the data scientist leading Project Aurora."`
Two facts in one sentence. The subject of the second is `Nila Rao` (an appositive),
not `The engineer`. The parse has to walk past the reduced relative `the data scientist
leading Project Aurora` and attribute the leading to the appositive head, or the graph
gains `(The engineer, leads, Project Aurora)` — which is false, and worse, is
*plausible*.

### 4.4 Overlapping resolutions (`RuleContext.resolution_for`)

Several resolution layers can cover the same span (ruler name, NER span, FastCoref
cluster). Returning the first match gave an answer that depended on layer order. The
resolution now overlaps on token indices and the highest-priority *non-overlapping*
candidate wins, which made the result order-independent.

### 4.5 Reduced-relative object fallback

After the subject trim, the object span may be empty. Rather than dropping the candidate,
the rule falls back to the object named explicitly in the clause, which is what keeps
`improves → Entity Resolution Engine` and `integrated_into → OrionEdge` intact.

## 5. Declined readings

The corpus is small enough to assert the *absence* of wrong readings, and the acceptance
tests do. All four of these must stay absent:

1. `(Nila Rao, leads, entity resolution work)` — the reporting-clause rejection.
2. `(Arun Mehta, uses, OrionEdge)` — `works on` is not `uses`.
3. `(Dr. Mira Sen, develops, OrionEdge)` — the developer's subject is Aether Analytics
   (`"The company develops OrionEdge"`), resolved via coreference to Mira's earlier
   `founded` sentence, not Mira as the grammatical subject.
4. Any `(PROJECT, works_at, ORGANIZATION)` — `works_at` requires employment language.

## 6. Literal nodes

`literal:graph_based_transaction_analysis` has `mention_count: 0` and is not in the entity
registry as a canonical entity. It enters the graph **only** as the object of
`focuses_on`.

This is deliberate, not a gap. The specification's entity list does not include it, and
the case-study graph shape expects it as a node. Giving it a mention count of 1 would
imply a declared entity that the registry does not contain. The UI therefore reports
**10 canonical entities** and **11 graph nodes**, and those two numbers are both correct.

## 7. What this document does not claim

- No rule is ML-learned. Every rule is a hand-written lexical/dependency pattern.
- No rule uses embeddings, a parser confidence threshold, or a probability. See §1.
- Extraction is not incremental or streaming; the corpus is processed as a whole. This is
  per spec §4 ("no scaling work") and would need rethinking before it was defensible at
  any real scale.