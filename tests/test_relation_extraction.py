"""S5/S6 acceptance: the relation and triple inventory the specification requires.

These run the whole corpus through coreference, relation rules and the triple builder, so
they are marked ``slow`` and ``acceptance``. Everything asserted here was transcribed from
specification sections 11 and 34 rather than from the pipeline's own output.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.acceptance]

#: (subject, relation, object) -- the 92 required triples.
REQUIRED_TRIPLES = frozenset(
    {
        ('Aether Analytics', 'develops', 'OrionEdge'),
        ('Aether Analytics', 'improves', 'GraphMatch'),
        ('Aether Analytics', 'partners_with', 'LedgerLine'),
        ('Aether Analytics', 'partners_with', 'Quantum Forge'),
        ('Aether Analytics', 'uses', 'ChatGPT'),
        ('AlphaFold', 'focuses_on', 'protein structure prediction'),
        ('Arun Mehta', 'collaborates_with', 'Nila Rao'),
        ('Arun Mehta', 'works_at', 'Aether Analytics'),
        ('Arun Mehta', 'works_on', 'OrionEdge'),
        ('CERN', 'develops', 'Large Hadron Collider'),
        ('CERN', 'partners_with', 'Fermilab'),
        ('ChatGPT', 'focuses_on', 'conversational AI assistants'),
        ('Christopher Nolan', 'develops', 'Inception'),
        ('DeepMind', 'develops', 'AlphaFold'),
        ('Demis Hassabis', 'founded', 'DeepMind'),
        ('Demis Hassabis', 'leads', 'DeepMind'),
        ('Denis Villeneuve', 'works_on', 'Dune'),
        ('Dmitri Volkov', 'collaborates_with', 'Priya Nair'),
        ('Dmitri Volkov', 'works_at', 'LedgerLine'),
        ('Dmitri Volkov', 'works_on', 'AuditLens'),
        ('Dr. Mira Sen', 'founded', 'Aether Analytics'),
        ('Dr. Mira Sen', 'leads', 'Project Aurora'),
        ('Elon Musk', 'founded', 'SpaceX'),
        ('Elon Musk', 'leads', 'SpaceX'),
        ('European Commission', 'focuses_on', 'trade policy'),
        ('FIFA', 'partners_with', 'adidas'),
        ('FIFA', 'uses', 'World Cup'),
        ('Fermilab', 'works_on', 'Large Hadron Collider'),
        ('Firefox', 'focuses_on', 'web browsing'),
        ('Gianni Infantino', 'leads', 'World Cup'),
        ('Google', 'uses', 'AlphaFold'),
        ('GraphMatch', 'deployed_at', 'Helios Bank'),
        ('GraphMatch', 'focuses_on', 'ledger anomaly detection'),
        ('GraphMatch', 'uses', 'Python'),
        ('Guido van Rossum', 'develops', 'Python'),
        ('Guido van Rossum', 'works_on', 'Python'),
        ('Hayao Miyazaki', 'founded', 'Studio Ghibli'),
        ('Hayao Miyazaki', 'leads', 'Studio Ghibli'),
        ('Helios Bank', 'customer_of', 'Aether Analytics'),
        ('Helios Bank', 'uses', 'Firefox'),
        ('Helios Bank', 'uses', 'OrionEdge'),
        ('Inception', 'focuses_on', 'dream heist thrillers'),
        ('Jens Stoltenberg', 'leads', 'NATO'),
        ('Kyoto Animation', 'collaborates_with', 'Makoto Shinkai'),
        ('Large Hadron Collider', 'focuses_on', 'particle physics experiments'),
        ('LedgerLine', 'integrated_into', 'OrionEdge'),
        ('Linus Torvalds', 'leads', 'Linux Foundation'),
        ('Linus Torvalds', 'works_on', 'Linux'),
        ('Linux', 'focuses_on', 'open source kernels'),
        ('Linux Foundation', 'develops', 'Linux'),
        ('Makoto Shinkai', 'develops', 'Your Name'),
        ('Makoto Shinkai', 'works_on', 'Suzume'),
        ('Marc Andreessen', 'founded', 'Netscape'),
        ('Marc Andreessen', 'leads', 'Netscape'),
        ('Mario', 'focuses_on', 'platforming games'),
        ('Microsoft', 'uses', 'ChatGPT'),
        ('Mozilla', 'develops', 'Firefox'),
        ('Mozilla', 'uses', 'Linux'),
        ('NASA', 'partners_with', 'SpaceX'),
        ('NASA', 'uses', 'Starship'),
        ('NATO', 'partners_with', 'European Union'),
        ('Netscape', 'partners_with', 'Mozilla'),
        ('Nila Rao', 'founded', 'Resolution Working Group'),
        ('Nila Rao', 'improves', 'Entity Resolution Engine'),
        ('Nila Rao', 'leads', 'Project Aurora'),
        ('Nintendo', 'develops', 'Mario'),
        ('Nintendo', 'partners_with', 'DeNA'),
        ('OpenAI', 'develops', 'ChatGPT'),
        ('OrionEdge', 'deployed_at', 'Helios Bank'),
        ('OrionEdge', 'uses', 'Linux'),
        ('Priya Nair', 'collaborates_with', 'Nila Rao'),
        ('Priya Nair', 'leads', 'GraphMatch'),
        ('Priya Nair', 'leads', 'Resolution Working Group'),
        ('Project Aurora', 'focuses_on', 'graph-based transaction analysis'),
        ('Project Aurora', 'integrated_into', 'OrionEdge'),
        ('Python', 'focuses_on', 'general purpose programming'),
        ('Python Software Foundation', 'develops', 'Python'),
        ('Resolution Working Group', 'uses', 'GraphMatch'),
        ('Sam Altman', 'founded', 'OpenAI'),
        ('Sam Altman', 'leads', 'OpenAI'),
        ('Shigeru Miyamoto', 'develops', 'Mario'),
        ('SpaceX', 'develops', 'Starship'),
        ('Spirited Away', 'focuses_on', 'animated fantasy storytelling'),
        ('Spotify', 'focuses_on', 'music streaming'),
        ('Starship', 'focuses_on', 'deep space transport'),
        ('Studio Ghibli', 'develops', 'Spirited Away'),
        ('Suzume', 'focuses_on', 'animated disaster aftermath'),
        ('Universal Music', 'collaborates_with', 'Daniel Ek'),
        ('Ursula von der Leyen', 'leads', 'European Commission'),
        ('Warner Bros', 'develops', 'Dune'),
        ('World Cup', 'focuses_on', 'international football'),
        ('Your Name', 'focuses_on', 'body swap romance'),
    }
)

#: Optional relations, enabled by default. Both are `contains` from a non-canonical cue.
OPTIONAL_TRIPLES = frozenset(
    {
        ("Project Aurora", "contains", "Entity Resolution Engine"),
        ("Project Aurora", "contains", "Graph Matching Module"),
    }
)

#: (document_id, subject, relation, object) that must never appear.
#:
#: Each one is a specific misreading, so the test names the failure it guards against
#: rather than just asserting an absence.
FORBIDDEN_TRIPLES = {
    ("doc_05_customer", "Arun Mehta", "reuses", "Graph Matching Module"):
        "reported speech is belief, not an assertion of action",
    ("doc_04_aurora", "She", "reduces", "duplicate alerts"):
        "'duplicate alerts' is a LITERAL, not an entity, so no triple should exist",
    ("doc_02_team", "Nila Rao", "leads", "entity resolution work"):
        "'the entity resolution work' is a common noun phrase, not Project Aurora",
    ("doc_05_customer", "The company", "customer_of", "Helios Bank"):
        "FastCoref mis-binds 'The company' to Helios Bank; the declared lexicon says "
        "Aether Analytics",
}


@pytest.fixture(scope="module")
def triples(extraction_result):
    return extraction_result.triples


@pytest.fixture(scope="module")
def extracted(extraction_result) -> set[tuple[str, str, str]]:
    return {t.as_tuple for t in extraction_result.triples}


@pytest.mark.parametrize("want", sorted(REQUIRED_TRIPLES))
def test_required_triple_present(extracted, want):
    assert want in extracted


@pytest.mark.parametrize("want", sorted(OPTIONAL_TRIPLES))
def test_optional_triple_present(extracted, want):
    assert want in extracted


def test_no_unexpected_triples(extracted):
    """Exactly 61 edges: the inventory is closed, so extras are errors, not bonus."""
    unexpected = extracted - REQUIRED_TRIPLES - OPTIONAL_TRIPLES
    assert not unexpected, f"unexpected triples: {sorted(unexpected)}"


def test_triple_count(triples):
    assert len(triples) == 94
    assert len({t.as_tuple for t in triples}) == 94, "duplicates should be deduplicated"


@pytest.mark.parametrize("key, reason", sorted(FORBIDDEN_TRIPLES.items()))
def test_forbidden_triple_absent(triples, key, reason):
    doc_id, subject, relation, obj = key
    offenders = [
        t
        for t in triples
        if (t.document_id, t.subject, t.relation.value, t.object) == key
    ]
    assert not offenders, f"{reason}"


class TestTripleMetadata:
    def test_every_triple_records_provenance(self, triples):
        for t in triples:
            assert t.document_id, f"{t.as_tuple} has no document"
            assert t.sentence_index >= 0
            assert t.original_subject, f"{t.as_tuple} has no source surface for its subject"
            assert t.original_object

    def test_confidence_bands_match_specification(self, triples):
        """Only the inferred relation may sit in the 0.80 band.

        A coreference-dependent triple scored 1.00 would misreport how it was derived.
        """
        for t in triples:
            if t.inferred:
                assert float(t.confidence) == 0.80, f"{t.as_tuple} is inferred"
            elif t.coref_resolved:
                assert float(t.confidence) == 0.90, f"{t.as_tuple} used coreference"
            else:
                assert float(t.confidence) == 1.00, f"{t.as_tuple} is direct"

    def test_customer_of_is_the_only_inferred_triple(self, triples):
        inferred = {t.as_tuple for t in triples if t.inferred}
        assert inferred == {("Helios Bank", "customer_of", "Aether Analytics")}

    def test_coref_flagged_triples_use_an_anaphoric_surface(self, triples):
        """A `coref_resolved` triple must show a reference, not a proper name."""
        named_surfaces = {
            "Dr. Mira Sen", "Aether Analytics", "OrionEdge", "Project Aurora",
            "Arun Mehta", "Nila Rao", "Quantum Forge", "Helios Bank",
            "Entity Resolution Engine", "Graph Matching Module",
        }
        for t in triples:
            if t.coref_resolved:
                assert t.original_subject not in named_surfaces or (
                    t.original_object not in named_surfaces
                ), f"{t.as_tuple} is flagged coref but both surfaces are proper names"


class TestGraphInvariants:
    """Checks the specification places on the triple set as a whole."""

    def test_node_inventory(self, triples, registry):
        labels = {t.subject for t in triples} | {t.object for t in triples}
        expected = {e.label for e in registry.entities} | {e.label for e in registry.literals}
        assert labels == expected, (
            f"only declared labels should be reachable: missing {expected - labels}, "
            f"unexpected {labels - expected}"
        )

    def test_node_count(self, triples):
        labels = {t.subject for t in triples} | {t.object for t in triples}
        assert len(labels) == 86

    def test_type_distribution(self, triples, registry):
        """Twenty people, twenty-seven organisations, sixteen products, four projects,
        two components, seventeen literal objects."""
        from collections import Counter

        def type_of(label: str) -> str:
            for entity in registry.entities:
                if entity.label == label:
                    return entity.type.value.lower()
            return "literal"

        counts = Counter(
            type_of(label)
            for label in {t.subject for t in triples} | {t.object for t in triples}
        )
        assert counts == {
            "person": 20,
            "organization": 27,
            "product": 16,
            "project": 4,
            "component": 2,
            "literal": 17,
        }

    def test_no_self_loops(self, triples):
        assert [t.as_tuple for t in triples if t.subject == t.object] == []

    def test_all_endpoints_are_declared(self, triples, registry):
        known = {e.entity_id for e in registry.entities} | {
            e.entity_id for e in registry.literals
        }
        for t in triples:
            assert t.subject_id in known
            assert t.object_id in known


class TestOptionalRelationToggle:
    def test_optional_relations_can_be_disabled(self, config, registry, processor, documents):
        """`contains` is opt-out, so disabling it must drop exactly the 2 optional edges."""
        from src.coreference import (
            CoreferenceResolver,
            DeterministicCorefFallback,
            FastCorefResolver,
        )
        from src.document_loader import DocumentLoader  # noqa: F401 - documents come from fixture
        from src.rules import RelationExtractor
        from src.triple_builder import TripleBuilder

        resolver = CoreferenceResolver(
            config,
            registry,
            FastCorefResolver(config, processor.load()),
            DeterministicCorefFallback(registry),
        )
        pairs = []
        for document in documents:
            processed = processor.process(document)
            index = {(r.start_char, r.end_char): r for r in resolver.resolve(processed)}
            pairs.append((processed, index))

        extractor = RelationExtractor(enable_optional_relations=False)
        candidates, _ = extractor.extract(pairs, registry=registry)
        triples, _ = TripleBuilder(registry).build(candidates)
        got = {t.as_tuple for t in triples}
        assert got == REQUIRED_TRIPLES
        assert not (got & OPTIONAL_TRIPLES)
