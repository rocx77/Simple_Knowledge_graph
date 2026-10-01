"""Stage 2 -- spaCy processing and mention detection.

Produces the linguistic backbone the rest of the pipeline depends on: tokens with POS
tags and dependency labels, sentences, NER spans, noun chunks, and a unified list of
entity mentions.

Two design decisions are load-bearing and both come from measurement, not assumption:

1. **The EntityRuler must be registered after ``ner``.** spaCy's generic model
   mislabels the fictional corpus badly: "Aether Analytics" comes back as ``PERSON``,
   "Nila Rao" as ``ORG``, "Project Aurora" as ``PERSON``. With the ruler last and
   ``overwrite_ents`` enabled, the declared lexicon wins.

2. **Definite descriptions and pronouns are deliberately NOT ruler patterns.**
   ``the bank``, ``the platform``, ``she`` are handled by coreference (stage 3).
   Putting them in the ruler would manufacture entities that have no canonical
   identity and would corrupt the NER layer.

The spaCy ``Language`` object is expensive and is loaded once here and shared with
FastCoref, which needs one for candidate span detection (rule P1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

from .config import AppConfig
from .errors import ModelUnavailableError
from .logging_utils import get_logger
from .models import Document, EntityMention, MentionSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
else:  # runtime import: Token/Span are needed for isinstance checks in _make_mention
    from spacy.tokens import Span, Token

logger = get_logger(__name__)

#: Head nouns whose noun chunk denotes an entity even though NER will not tag it.
#: This is the mechanism that lets "The company develops OrionEdge" become
#: "(Aether Analytics, develops, OrionEdge)".
COMMON_NOUN_HEADS = frozenset(
    {"company", "bank", "platform", "project", "engineer", "extension", "engine", "team"}
)

#: Pronoun lemmas treated as anaphoric candidates for coreference.
PRONOUN_LEMMAS = frozenset(
    {"he", "she", "it", "they", "his", "her", "its", "their", "them", "him"}
)

#: spaCy NER labels that never denote an entity in this domain.
_NON_ENTITY_NER_LABELS = frozenset(
    {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}
)

#: Higher wins when two mention sources claim overlapping spans.
_SOURCE_PRIORITY = {
    MentionSource.RULER: 0,
    MentionSource.SPACY_NER: 1,
    MentionSource.COMMON_NOUN: 2,
    MentionSource.PRONOUN: 3,
    MentionSource.COREF: 4,
}


def _morph_value(token: "Token") -> str:  # noqa: F821 - spacy Token
    """Return a token's morphological features as a ``key=value|key=value`` string.

    ``MorphAnalysis.get()`` returns a *list* (e.g. ``['Def']``), so any comparison
    against a bare string silently fails. This normalises to the first value.
    """
    if not token.morph:
        return ""
    return "|".join(f"{key}={values[0] if values else ''}" for key, values in sorted(token.morph))


def _morph_feature(token: "Token", key: str) -> str:  # noqa: F821 - spacy Token
    """Return one morphological feature as a plain string, or ``""`` if absent."""
    values = token.morph.get(key)
    if not values:
        return ""
    first = values[0] if isinstance(values, (list, tuple)) else values
    return str(first).strip().lower()


def _is_definite_reference(chunk: "Span") -> bool:  # noqa: F821 - spacy Span
    """True when a noun chunk is a definite description, i.e. an anaphoric reference.

    Definiteness is the signal that separates a reference from a first mention:
    ``the engineer`` refers back to Arun Mehta, whereas ``a senior machine learning
    engineer`` introduces a job title and must not become an entity mention.
    """
    first = chunk[0]
    if first.pos_ == "DET" and _morph_feature(first, "Definite") == "def":
        return True
    return first.pos_ == "PRON"


@dataclass(frozen=True, slots=True)
class RulerPattern:
    """One EntityRuler pattern.

    Invariants: ``surface`` is non-empty and ``entity_type`` is one of the types in the
    canonical registry.
    """

    surface: str
    entity_type: str
    canonical_label: str
    entity_id: str = ""


@dataclass(frozen=True, slots=True)
class TokenView:
    """A spaCy token flattened for debugging and explanation (specification section 6).

    Retains exactly the fields the specification asks to preserve, plus the lemma and
    morphological features that later stages need for gender agreement.
    """

    index: int
    text: str
    pos: str
    dep: str
    head_text: str
    head_index: int
    lemma: str
    ent_type: str
    start_char: int
    end_char: int
    is_entity_token: bool = False
    gender: str = ""

    @classmethod
    def from_token(cls, token: "Token") -> "TokenView":  # noqa: F821 - spacy Token
        return cls(
            index=token.i,
            text=token.text,
            pos=token.pos_,
            dep=token.dep_,
            head_text=token.head.text if token.head is not token else "<ROOT>",
            head_index=token.head.i,
            lemma=token.lemma_,
            ent_type=token.ent_type_,
            # spaCy exposes the start offset as `.idx`; Token has no `start_char`.
            start_char=token.idx,
            end_char=token.idx + len(token.text),
            is_entity_token=token.ent_type_ != "",
            gender=_morph_feature(token, "Gender"),
        )


@dataclass(frozen=True, slots=True)
class SentenceView:
    """A sentence with its char span, kept so evidence can quote an exact location."""

    index: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class NounChunkView:
    """A noun chunk flattened to text plus char span (used to expand rule objects)."""

    text: str
    start_char: int
    end_char: int
    root_index: int


@dataclass
class ProcessedDocument:
    """One document after spaCy processing.

    Holds the live spaCy ``Doc`` (needed by the rules, which inspect real dependency
    trees) alongside flattened views used by the UI and the artifacts.

    Invariants:
        * ``sentences`` are ordered and non-overlapping.
        * ``mentions`` are ordered by ``start_char`` and never overlap.
        * ``mentions_by_sentence`` indexes the same objects, not copies.
    """

    document: Document
    doc: "Doc"
    sentences: tuple[SentenceView, ...] = ()
    tokens: tuple[TokenView, ...] = ()
    noun_chunks: tuple[NounChunkView, ...] = ()
    mentions: tuple[EntityMention, ...] = ()
    ruler_surface_spans: frozenset[tuple[int, int]] = frozenset()
    mentions_by_sentence: dict[int, tuple[EntityMention, ...]] = field(default_factory=dict)

    def sentence_text(self, index: int) -> str:
        return self.sentences[index].text if 0 <= index < len(self.sentences) else ""

    def mention_at(self, start_char: int) -> EntityMention | None:
        return next((m for m in self.mentions if m.start_char == start_char), None)

    def mentions_in_sentence(self, index: int) -> tuple[EntityMention, ...]:
        return self.mentions_by_sentence.get(index, ())


class NLPProcessor:
    """Owns the spaCy ``Language`` object and turns documents into processed views."""

    def __init__(self, config: AppConfig, ruler_patterns: Sequence[RulerPattern] = ()) -> None:
        self._config = config
        self._patterns = tuple(ruler_patterns)
        self._nlp: "Language | None" = None
        self._ruler_surface_spans: frozenset[tuple[int, int]] = frozenset()

    # -- model loading (rule P1: load once, reuse) --------------------------

    def load(self) -> "Language":
        """Load spaCy and register the EntityRuler. Cached on the instance."""
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - spaCy is a hard dependency
            raise ModelUnavailableError("spaCy is not installed.") from exc

        try:
            nlp = spacy.load(self._config.spacy_model)
        except OSError as exc:
            raise ModelUnavailableError(
                f"Could not load spaCy model {self._config.spacy_model!r}.\n"
                f"  Install it with:\n"
                f"  .venv\\Scripts\\python.exe -m pip install "
                f"https://github.com/explosion/spacy-models/releases/download/"
                f"{self._config.spacy_model}-3.8.0/{self._config.spacy_model}-3.8.0-py3-none-any.whl\n"
                f"  Run 'python scripts/verify_setup.py' to re-check the environment."
            ) from exc

        if self._patterns:
            self._add_entity_ruler(nlp)
            self._ruler_surface_spans = frozenset(
                (p.surface.lower(), p.canonical_label) for p in self._patterns
            )

        logger.info(
            "spaCy model %s loaded (pipes=%s, ruler patterns=%d)",
            self._config.spacy_model,
            ",".join(nlp.pipe_names),
            len(self._patterns),
        )
        self._nlp = nlp
        return nlp

    def _add_entity_ruler(self, nlp: "Language") -> None:
        """Register the EntityRuler *after* ``ner`` so it overwrites spaCy's labels."""
        if "entity_ruler" in nlp.pipe_names:
            ruler = nlp.get_pipe("entity_ruler")
        else:
            # No explicit position: add_pipe appends, which puts the ruler after ner.
            ruler = nlp.add_pipe(
                "entity_ruler",
                config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"},
            )

        # Must go through `add_patterns` so the ruler registers both the patterns and
        # their tokenised forms. Writing straight into `ruler.phrase_matcher` leaves
        # `ruler.patterns` empty and spaCy then warns that the component has no
        # patterns defined.
        #
        # The NER label is the entity *type*, never the canonical name, so the spaCy
        # entity layer stays semantically meaningful and the registry owns naming.
        ruler.add_patterns(
            [
                {"label": pattern.entity_type, "pattern": pattern.surface}
                for pattern in sorted(self._patterns, key=lambda p: -len(p.surface))
            ]
        )

    # -- processing ---------------------------------------------------------

    def process(self, document: Document) -> ProcessedDocument:
        """Run spaCy over one document and build its views and mentions."""
        nlp = self.load()
        doc = nlp(document.text)

        sentences = tuple(
            SentenceView(
                index=i,
                text=sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char,
            )
            for i, sent in enumerate(doc.sents)
        )
        tokens = tuple(TokenView.from_token(tok) for tok in doc)
        chunks = tuple(
            NounChunkView(
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                root_index=chunk.root.i,
            )
            for chunk in doc.noun_chunks
        )

        ruler_spans = self._ruler_span_lookup(document.text, doc)
        mentions = self._detect_mentions(document, doc, ruler_spans)

        by_sentence: dict[int, tuple[EntityMention, ...]] = {}
        for mention in mentions:
            by_sentence.setdefault(mention.sentence_index, ())
            by_sentence[mention.sentence_index] += (mention,)

        logger.debug(
            "%s: %d sentence(s), %d token(s), %d noun chunk(s), %d mention(s)",
            document.filename, len(sentences), len(tokens), len(chunks), len(mentions),
        )

        return ProcessedDocument(
            document=document,
            doc=doc,
            sentences=sentences,
            tokens=tokens,
            noun_chunks=chunks,
            mentions=mentions,
            ruler_surface_spans=ruler_spans,
            mentions_by_sentence=by_sentence,
        )

    def _ruler_span_lookup(self, text: str, doc: "Doc") -> dict[int, str]:
        """Map ``start_char -> canonical_label`` for spans produced by the ruler.

        A span is attributed to the ruler when it matches a declared surface form, which
        is what tells the resolver it may trust the label over spaCy's.
        """
        surfaces = {p.surface.lower() for p in self._patterns}
        lookup: dict[int, str] = {}
        for ent in doc.ents:
            if ent.text.lower() in surfaces:
                lookup[ent.start_char] = ent.text
        return lookup

    def _detect_mentions(
        self, document: Document, doc: "Doc", ruler_spans: dict[int, str]
    ) -> tuple[EntityMention, ...]:
        """Collect candidate mentions from four sources and resolve overlaps.

        Overlaps are resolved by source priority (ruler beats NER beats common noun
        beats pronoun) and then by span length, so "Project Aurora" is never truncated
        to "Aurora" by a nested span.
        """
        candidates: list[EntityMention] = []

        # 1 + 2. NER and ruler spans.
        for ent in doc.ents:
            if ent.label_ in _NON_ENTITY_NER_LABELS:
                continue  # quantities, dates and ordinals are never entity mentions
            source = (
                MentionSource.RULER if ent.start_char in ruler_spans else MentionSource.SPACY_NER
            )
            candidates.append(
                self._make_mention(document, doc, ent, source, label=ent.label_)
            )

        # 3. Definite common-noun noun chunks.
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "appos":
                # An appositive describes the entity it is attached to; it is not an
                # independent reference to it. Without this, "a real-time fraud
                # detection platform" (the appositive describing OrionEdge in doc_01)
                # would be conflated with "the platform" (a genuine reference to
                # OrionEdge in doc_03).
                continue
            if chunk.root.lemma_.lower() not in COMMON_NOUN_HEADS:
                continue
            if not _is_definite_reference(chunk):
                # Anaphoric references are definite. "a senior machine learning
                # engineer" is a job title, not a reference to Arun Mehta.
                continue
            candidates.append(
                self._make_mention(document, doc, chunk, MentionSource.COMMON_NOUN, label="")
            )

        # 4. Pronouns.
        for token in doc:
            if token.pos_ == "PRON" and token.lemma_.lower() in PRONOUN_LEMMAS:
                candidates.append(
                    self._make_mention(document, doc, token, MentionSource.PRONOUN, label="")
                )

        return tuple(self._resolve_overlaps(candidates))

    def _make_mention(
        self,
        document: Document,
        doc: "Doc",
        span: "Span | Token",  # noqa: F821 - spacy Span or Token
        source: MentionSource,
        *,
        label: str,
    ) -> EntityMention:
        # Span exposes `.start`/`.root`/`.start_char`; Token exposes only `.i`/`.idx`.
        is_span = isinstance(span, Span)
        start_token = span.start if is_span else span.i
        root = span.root if is_span else span
        start_char = span.start_char if is_span else span.idx
        end_char = span.end_char if is_span else span.idx + len(span.text)
        sentence_index = self._sentence_index_of(doc, start_token)
        return EntityMention(
            text=span.text,
            start_char=start_char,
            end_char=end_char,
            label=label or root.lemma_,
            sentence_index=sentence_index,
            document_id=document.document_id,
            token_index=start_token,
            root_index=root.i,
            source=source,
        )

    @staticmethod
    def _sentence_index_of(doc: "Doc", token_index: int) -> int:
        for index, sent in enumerate(doc.sents):
            if sent.start <= token_index < sent.end:
                return index
        return 0

    @staticmethod
    def _resolve_overlaps(candidates: Iterable[EntityMention]) -> list[EntityMention]:
        """Keep the highest-priority mention for any overlapping region.

        Priority is source first, then span length. Sorting by priority *before*
        selecting matters: "the Entity Resolution Engine" (a common-noun chunk) starts
        earlier than the ruler span "Entity Resolution Engine" inside it, so a plain
        left-to-right scan would keep the chunk and discard the correctly typed ruler
        match.
        """
        ordered = sorted(
            candidates,
            key=lambda m: (
                _SOURCE_PRIORITY.get(m.source, 9),
                -(m.end_char - m.start_char),
                m.start_char,
            ),
        )
        kept: list[EntityMention] = []
        for mention in ordered:
            overlaps = any(
                not (mention.end_char <= k.start_char or mention.start_char >= k.end_char)
                for k in kept
            )
            if not overlaps:
                kept.append(mention)
        kept.sort(key=lambda m: (m.start_char, m.end_char))
        return kept


__all__ = [
    "NLPProcessor",
    "ProcessedDocument",
    "RulerPattern",
    "SentenceView",
    "TokenView",
    "NounChunkView",
    "COMMON_NOUN_HEADS",
    "PRONOUN_LEMMAS",
]