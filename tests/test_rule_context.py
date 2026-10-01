"""Unit tests for span matching and parse-shape assumptions in :mod:`src.rules.base`.

Every test here encodes a bug that was actually hit and fixed during Phase 4c. Each
docstring states the trap, because the failures were all silent (wrong output) or
non-terminating (identity comparisons on spaCy wrappers) rather than loud exceptions.
"""

from __future__ import annotations

import pytest

from src.entity_resolver import EntityRegistry
from src.models import EntityType, MentionSource
from src.nlp_processor import NLPProcessor

from .conftest import make_context

AETHER = "organization:aether_analytics"
ORIONEDGE = "product:orionedge"
AURORA = "project:project_aurora"
NILA = "person:nila_rao"
RAO = "person:arun_mehta"
ENGINE = "component:entity_resolution_engine"


def _spec(surface, entity_id, label, source=MentionSource.RULER):
    return (surface, entity_id, label, source)


@pytest.fixture(scope="module")
def ctx_factory(processor: NLPProcessor, registry: EntityRegistry):
    """Build a RuleContext for arbitrary text with hand-written resolutions."""

    def build(text, specs, sentence_index=0):
        return make_context(text, specs, processor, registry, sentence_index)

    return build


# ---------------------------------------------------------------------------
# spaCy wrapper identity
# ---------------------------------------------------------------------------


class TestTokenWrapperIdentity:
    def test_token_is_its_own_head_only_at_root(self, ctx_factory):
        """`token.head is token` is false even for the ROOT token.

        spaCy builds a fresh Python wrapper on every attribute access, so an identity
        test for "am I the ROOT?" never succeeds. Getting this wrong is what made an
        ancestor walk in `is_under_reporting_verb` loop forever.
        """
        ctx = ctx_factory("Arun Mehta founded Aether Analytics.", ())
        root = ctx.sentence.root
        assert root.i == root.head.i, "index equality is the reliable root test"
        assert root.head is not root, "confirms identity is unreliable here"

    def test_ancestor_walk_terminates(self, ctx_factory):
        """The reporting-verb guard walks up the tree; it must return, not spin."""
        ctx = ctx_factory(
            "Arun Mehta said the extension will reuse Aurora's module.",
            (_spec("Arun Mehta", RAO, "Arun Mehta"),),
        )
        for token in ctx.sentence:
            ctx.is_under_reporting_verb(token)

    def test_noun_chunk_lookup_survives_wrapper_reaccess(self, ctx_factory):
        ctx = ctx_factory(
            "Nila Rao improves the Entity Resolution Engine.",
            (_spec("Nila Rao", NILA, "Nila Rao"),),
        )
        token = ctx.sentence[1]
        chunk = ctx.noun_chunk_of(token)
        assert chunk is not None
        assert chunk.root.i == token.i


# ---------------------------------------------------------------------------
# Span matching
# ---------------------------------------------------------------------------


class TestResolutionSpanMatching:
    def test_mention_nested_inside_requested_span(self, ctx_factory):
        """The noun chunk includes the determiner; the ruler mention does not.

        `the Entity Resolution Engine` is the chunk, but coreference resolved the inner
        span `Entity Resolution Engine`. Containment in that direction only would fail.
        """
        text = "Nila Rao improves the Entity Resolution Engine."
        ctx = ctx_factory(text, (_spec("Entity Resolution Engine", ENGINE, "Entity Resolution Engine"),))
        start = text.index("the Entity Resolution Engine")
        arg = ctx.argument_for_span(start, start + len("the Entity Resolution Engine"))
        assert arg is not None
        assert arg.entity_id == ENGINE

    def test_requested_span_nested_inside_mention(self, ctx_factory):
        """The mirror case: the possessor is a sub-span of one big noun chunk.

        `Aurora's graph matching module` is a single chunk, so a request for just the
        possessor `Aurora` sits inside a resolution that cannot contain it.
        """
        text = "Arun Mehta said the extension will reuse Aurora's graph matching module."
        specs = (
            _spec("Arun Mehta", RAO, "Arun Mehta"),
            _spec("graph matching module", "component:graph_matching_module",
                  "Graph Matching Module"),
            _spec("Aurora", AURORA, "Project Aurora"),
        )
        ctx = ctx_factory(text, specs)
        possessor = next(t for t in ctx.sentence if t.dep_ == "poss")
        owner = ctx.argument_for_own_token(possessor)
        assert owner is not None
        assert owner.entity_id == AURORA
        assert owner.surface == "Aurora"

    def test_tightest_overlap_wins(self, ctx_factory):
        """Overlapping mentions must select the specific one, not a neighbour."""
        text = "Nila Rao improves the Entity Resolution Engine."
        specs = (
            _spec("Entity Resolution Engine", ENGINE, "Entity Resolution Engine"),
            _spec("Engine", "component:other", "Other"),
        )
        ctx = ctx_factory(text, specs)
        start = text.index("the Entity Resolution Engine")
        arg = ctx.argument_for_span(start, start + len("the Entity Resolution Engine"))
        assert arg.entity_id == ENGINE, "the longer overlapping mention must win"

    def test_unrelated_span_resolves_to_nothing(self, ctx_factory):
        ctx = ctx_factory(
            "Nila Rao leads the entity resolution work.",
            (_spec("Nila Rao", NILA, "Nila Rao"),),
        )
        start = ctx.doc_text.index("the entity resolution work")
        assert ctx.argument_for_span(start, start + len("the entity resolution work")) is None


# ---------------------------------------------------------------------------
# Parse shape
# ---------------------------------------------------------------------------


class TestReducedRelativeAppositive:
    TEXT = "Nila Rao, the data scientist leading Project Aurora, improves the engine."

    def test_subject_is_appositive_hosts_head(self, ctx_factory):
        """`leading` is an `amod` on Project Aurora, which is an `appos` on Nila Rao.

        The subject is the appositive host's *head*. Searching the host's subtree finds
        nothing because Rao is an ancestor of Project Aurora, not a descendant.
        """
        ctx = ctx_factory(self.TEXT, (_spec("Nila Rao", NILA, "Nila Rao"),))
        verb = next(t for t in ctx.sentence if t.lemma_ == "lead")
        assert verb.dep_ == "amod"
        subjects = ctx.subject_tokens(verb)
        assert len(subjects) == 1
        # The NP head of "Nila Rao", not the first token of the chunk.
        assert ctx.argument_for_token(subjects[0]).entity_id == NILA

    def test_object_is_the_verbs_own_head(self, ctx_factory):
        """A reduced relative has no object *child*; the nominal is its head."""
        ctx = ctx_factory(self.TEXT, (_spec("Nila Rao", NILA, "Nila Rao"),))
        verb = next(t for t in ctx.sentence if t.lemma_ == "lead")
        assert ctx.direct_object(verb) is None
        assert verb.head.text == "Aurora"


class TestReportingClauseGuard:
    TEXT = "Arun Mehta said the extension will reuse Aurora's graph matching module."

    def test_reported_clause_is_flagged(self, ctx_factory):
        ctx = ctx_factory(self.TEXT, (_spec("Arun Mehta", RAO, "Arun Mehta"),))
        reuse = next(t for t in ctx.sentence if t.lemma_ == "reuse")
        assert ctx.is_under_reporting_verb(reuse)

    def test_asserted_clause_is_not_flagged(self, ctx_factory):
        text = "Arun Mehta reuses Aurora's graph matching module."
        ctx = ctx_factory(text, (_spec("Arun Mehta", RAO, "Arun Mehta"),))
        reuse = next(t for t in ctx.sentence if t.lemma_ == "reuse")
        assert not ctx.is_under_reporting_verb(reuse)


class TestLiteralEndpoints:
    def test_literal_allowed_only_when_requested(self, ctx_factory):

        text = "Project Aurora focuses on graph-based transaction analysis."
        ctx = ctx_factory(
            text,
            (
                _spec("Project Aurora", AURORA, "Project Aurora"),
                _spec("graph-based transaction analysis",
                      "literal:graph_based_transaction_analysis",
                      "graph-based transaction analysis"),
            ),
        )
        start = text.index("graph-based")
        span = (start, start + len("graph-based transaction analysis"))
        assert ctx.argument_for_span(*span, allow_literal=False) is None
        arg = ctx.argument_for_span(*span, allow_literal=True)
        assert arg is not None
        assert arg.entity_type is EntityType.LITERAL


class TestConfidenceBands:
    def test_coref_resolved_tracks_mention_kind_not_resolution_layer(self, ctx_factory):
        """A definite description resolved from the lexicon is still coreference.

        `coref_resolved` must come from how the mention was found, not from which layer
        resolved it -- otherwise "The company develops OrionEdge" is scored 1.00 instead
        of the specified 0.90, even though no model was involved.
        """
        from src.rules.base import confidence_for

        text = "The company develops OrionEdge."
        obj = _spec("OrionEdge", ORIONEDGE, "OrionEdge")

        # Same text, same lexicon hit, but the subject mention was found by the parser as
        # a common noun rather than by the ruler as a proper name.
        anaphoric = ctx_factory(
            text, (_spec("The company", AETHER, "Aether Analytics", MentionSource.COMMON_NOUN), obj)
        )
        named = ctx_factory(text, (_spec("The company", AETHER, "Aether Analytics"), obj))

        arg_anaphoric = anaphoric.argument_for_span(0, len("The company"))
        arg_named = named.argument_for_span(0, len("The company"))
        object_arg = anaphoric.argument_for_span(
            text.index("OrionEdge"), text.index("OrionEdge") + len("OrionEdge")
        )

        assert arg_anaphoric is not None and arg_named is not None and object_arg is not None
        assert arg_anaphoric.was_coref_resolved, "a definite description is a reference"
        assert not arg_named.was_coref_resolved, "a ruler hit names the entity outright"
        assert confidence_for(subject=arg_anaphoric, obj=object_arg) == 0.90
        assert confidence_for(subject=arg_named, obj=object_arg) == 1.00
        assert confidence_for(subject=arg_named, obj=object_arg, inferred=True) == 0.80

        # A purely named endpoint pair stays in the top band.
        both_named = ctx_factory(
            "Aether Analytics develops OrionEdge.",
            (_spec("Aether Analytics", AETHER, "Aether Analytics"), obj),
        )
        subj = both_named.argument_for_span(0, len("Aether Analytics"))
        assert subj is not None
        assert confidence_for(subject=subj, obj=object_arg) == 1.00
