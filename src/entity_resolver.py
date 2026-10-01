"""Stage 4 -- canonical entity registry and entity resolution.

Turns a surface mention into exactly one canonical entity, so that "OrionEdge",
"the platform" and "it" all become the single node ``product:orionedge``.

``data/entities.yaml`` is the declared **lexicon**: it says which names exist and how
they alias one another. It deliberately says nothing about relations -- the graph must
be derived from the documents (charter invariant I2, guideline M3).

Lookup precedence follows specification section 20:

    exact canonical name -> exact alias -> normalised lowercase -> possessive stripped

with **longest match first** at every level, so ``Project Aurora`` beats ``Aurora`` and
``Entity Resolution Engine`` is never truncated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .ids import make_entity_id, slugify
from .logging_utils import get_logger
from .models import CanonicalEntity, EntityMention, EntityType, LiteralNode
from .nlp_processor import RulerPattern

logger = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")

#: Pronoun surfaces are *ambiguous*: the same pronoun may denote different entities in
#: different documents ("She" is Dr. Mira Sen in one document and Nila Rao in another).
#: A static table therefore must never resolve them -- doing so silently assigns
#: whichever entity happened to be declared first. Pronouns are resolved by FastCoref,
#: or failing that by the nearest type-compatible antecedent.
PRONOUN_SURFACES = frozenset(
    {"he", "she", "it", "they", "him", "her", "his", "hers", "its", "their", "them", "himself", "herself"}
)


def normalise_surface(text: str) -> str:
    """Normalise a surface form for lookup: case-folded, whitespace collapsed."""
    return _WHITESPACE.sub(" ", text.strip().casefold())


def is_ambiguous_pronoun(text: str) -> bool:
    """True when a surface is a pronoun whose referent depends on context."""
    return normalise_surface(text) in PRONOUN_SURFACES


class EntityRegistry:
    """The canonical entity inventory, loaded from ``data/entities.yaml``.

    Invariants:
        * Every entity id matches ``"<type>:<slug>"`` and is unique.
        * ``_alias_index`` maps a normalised surface to at most one entity id. Building
          it longest-first means the longest key wins any collision.
        * ``_description_index`` contains only *unambiguous* anaphors. Pronouns are
          excluded by :data:`PRONOUN_SURFACES`.
    """

    def __init__(
        self,
        entries: Sequence[tuple[CanonicalEntity, tuple[str, ...]]],
        literals: Sequence[LiteralNode] = (),
    ) -> None:
        self._entries = tuple(entries)
        self._entities = tuple(entity for entity, _ in self._entries)
        self._literals = tuple(literals)
        self._by_id = {e.entity_id: e for e in self._entities}
        self._literal_by_id = {n.entity_id: n for n in self._literals}
        self._alias_index = self._build_alias_index(self._entities)
        self._description_index = self._build_description_index(self._entries)
        self._pronoun_index = self._build_pronoun_index(self._entries)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> EntityRegistry:
        """Load and validate the registry from YAML."""
        registry_path = path or (Path(__file__).resolve().parents[1] / "data" / "entities.yaml")
        if not registry_path.is_file():
            raise ConfigurationError(
                f"Entity registry not found at {registry_path}. "
                f"This file declares the canonical entity inventory and is required."
            )
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "PyYAML is required to read the entity registry. "
                "Install it with: .venv\\Scripts\\python.exe -m pip install pyyaml"
            ) from exc

        raw: dict[str, Any] = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        entries = [
            cls._build_entity(item, registry_path)
            for item in raw.get("entities", []) or []
        ]
        if not entries:
            raise ConfigurationError(f"No entities declared in {registry_path}")

        entities = [entity for entity, _anaphors in entries]
        literals = [
            cls._build_literal(item, registry_path)
            for item in raw.get("literals", []) or []
        ]
        logger.info("Loaded %d canonical entities and %d literals from %s",
                    len(entities), len(literals), registry_path.name)
        return cls(entries, literals)

    @staticmethod
    def _build_entity(item: dict[str, Any], source: Path) -> tuple[CanonicalEntity, tuple[str, ...]]:
        try:
            label = str(item["label"]).strip()
            type_value = EntityType(str(item["type"]).strip().upper())
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(f"Malformed entity entry in {source}: {item!r} ({exc})") from exc

        aliases = tuple(dict.fromkeys(str(a).strip() for a in item.get("aliases", []) or [] if a))
        anaphors = tuple(dict.fromkeys(str(a).strip().lower() for a in item.get("anaphors", []) or [] if a))
        declared_id = str(item.get("id") or "").strip()
        entity_id = declared_id or make_entity_id(type_value, label)
        expected_id = make_entity_id(type_value, label)
        if declared_id and declared_id != expected_id:
            raise ConfigurationError(
                f"Entity id {declared_id!r} does not match the derived id {expected_id!r} "
                f"for {label!r} in {source}"
            )

        return (
            CanonicalEntity(
                entity_id=entity_id,
                label=label,
                type=type_value,
                aliases=aliases,
                description=str(item.get("description", "")).strip(),
            ),
            anaphors,
        )

    @staticmethod
    def _build_literal(item: dict[str, Any], source: Path) -> LiteralNode:
        label = str(item.get("label", "")).strip()
        if not label:
            raise ConfigurationError(f"Literal entry without a label in {source}: {item!r}")
        declared_id = str(item.get("id") or "").strip() or make_entity_id(EntityType.LITERAL, label)
        return LiteralNode(
            entity_id=declared_id,
            label=label,
            description=str(item.get("description", "")).strip(),
        )

    @staticmethod
    def _build_alias_index(entities: Iterable[CanonicalEntity]) -> dict[str, str]:
        """Map normalised alias -> entity id. Longest key is inserted first so that a
        shorter alias can never overwrite a more specific one."""
        index: dict[str, str] = {}
        pairs: list[tuple[str, str]] = []
        for entity in entities:
            for surface in entity.aliases:
                pairs.append((normalise_surface(surface), entity.entity_id))
            pairs.append((normalise_surface(entity.label), entity.entity_id))
        for surface, entity_id in sorted(pairs, key=lambda p: -len(p[0])):
            index.setdefault(surface, entity_id)
        return index

    @staticmethod
    def _build_description_index(
        entries: Sequence[tuple[CanonicalEntity, tuple[str, ...]]],
    ) -> dict[str, str]:
        """Map an unambiguous anaphor -> entity id. Pronouns are skipped.

        A key declared by two entities is dropped rather than resolved to whichever
        came first: an ambiguous description must reach the antecedent rule, where the
        document context decides.
        """
        claims: dict[str, set[str]] = {}
        for entity, anaphors in entries:
            for anaphor in anaphors:
                if is_ambiguous_pronoun(anaphor):
                    continue
                claims.setdefault(anaphor, set()).add(entity.entity_id)
        return {
            anaphor: next(iter(ids))
            for anaphor, ids in claims.items()
            if len(ids) == 1
        }

    @staticmethod
    def _build_pronoun_index(
        entries: Sequence[tuple[CanonicalEntity, tuple[str, ...]]],
    ) -> dict[str, tuple[str, ...]]:
        """Map a pronoun -> every entity it could denote. Diagnostics only."""
        claims: dict[str, list[str]] = {}
        for entity, anaphors in entries:
            for anaphor in anaphors:
                if is_ambiguous_pronoun(anaphor) and entity.entity_id not in claims.setdefault(anaphor, []):
                    claims[anaphor].append(entity.entity_id)
        return {anaphor: tuple(ids) for anaphor, ids in claims.items()}

    # -- accessors ----------------------------------------------------------

    @property
    def entities(self) -> tuple[CanonicalEntity, ...]:
        return self._entities

    @property
    def literals(self) -> tuple[LiteralNode, ...]:
        return self._literals

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(e.entity_id for e in self._entities)

    def entity(self, entity_id: str) -> CanonicalEntity | None:
        return self._by_id.get(entity_id)

    def literal(self, entity_id: str) -> LiteralNode | None:
        return self._literal_by_id.get(entity_id)

    def record(self, entity_id: str) -> CanonicalEntity | LiteralNode | None:
        """Either kind of node, canonical entity or literal."""
        return self._by_id.get(entity_id) or self._literal_by_id.get(entity_id)

    def label_of(self, entity_id: str) -> str:
        record = self.record(entity_id)
        return record.label if record is not None else entity_id

    def type_of(self, entity_id: str) -> EntityType | None:
        """Entity type of a canonical id, including LITERAL nodes."""
        entity = self._by_id.get(entity_id)
        if entity is not None:
            return entity.type
        return EntityType.LITERAL if entity_id in self._literal_by_id else None

    def anaphor_target(self, surface: str) -> str | None:
        """Resolve an *unambiguous* definite description to an entity id.

        Returns ``None`` for pronouns (which are context dependent) and for any
        description claimed by more than one entity.
        """
        if is_ambiguous_pronoun(surface):
            return None
        return self._description_index.get(normalise_surface(surface))

    def pronoun_candidates(self, surface: str) -> tuple[str, ...]:
        """Every entity a pronoun could denote. Used for diagnostics, never for output."""
        return self._pronoun_index.get(normalise_surface(surface), ())

    def anaphors_of(self, entity_id: str) -> tuple[str, ...]:
        for entity, anaphors in self._entries:
            if entity.entity_id == entity_id:
                return anaphors
        return ()

    def aliases_of(self, entity_id: str) -> tuple[str, ...]:
        entity = self._by_id.get(entity_id)
        return entity.aliases if entity else ()

    def is_literal(self, entity_id: str) -> bool:
        return entity_id in self._literal_by_id

    def resolve_literal(self, text: str) -> str | None:
        """Resolve a noun phrase to a declared literal node.

        Literals have no canonical spelling of their own beyond the declared label, so
        the comparison is exact after whitespace normalisation. There is deliberately no
        fuzzy matching here: an unmatched literal must produce no triple rather than a
        near-miss.
        """
        candidate = normalise_surface(text)
        if not candidate:
            return None
        for literal in self._literals:
            if normalise_surface(literal.label) == candidate:
                return literal.entity_id
        return None

    # -- lookup -------------------------------------------------------------

    def resolve_surface(self, text: str) -> str | None:
        """Resolve a surface form to a canonical entity id.

        Precedence (specification section 20):
        1. exact canonical name (case-insensitive),
        2. exact alias,
        3. possessive-stripped form, so ``Aurora's`` resolves to Project Aurora.
        """
        if not text:
            return None
        candidate = normalise_surface(text)
        if (found := self._alias_index.get(candidate)) is not None:
            return found

        stripped = normalise_surface(re.sub(r"'s$", "", text.strip(), flags=re.IGNORECASE))
        if stripped != candidate and (found := self._alias_index.get(stripped)) is not None:
            return found

        # Last resort: an unambiguous anaphor alias such as "the platform". Pronouns
        # are deliberately excluded -- they are context dependent.
        return self.anaphor_target(candidate)

    def find_in_text(self, text: str) -> tuple[str, str] | None:
        """Longest entity surface form occurring inside ``text``.

        Returns ``(entity_id, matched_surface)`` or ``None``. Used by the query parser,
        where the entity is embedded in a natural-language question rather than being
        the whole string.
        """
        lowered = normalise_surface(text)
        best: tuple[int, str, str] | None = None  # (length, entity_id, surface)
        for surface, entity_id in self._alias_index.items():
            index = lowered.find(surface)
            if index < 0:
                continue
            # Require a word boundary so "Aurora" does not match inside "AuroraBase".
            before_ok = index == 0 or not lowered[index - 1].isalnum()
            after = index + len(surface)
            after_ok = after >= len(lowered) or not lowered[after].isalnum()
            if not (before_ok and after_ok):
                continue
            if best is None or len(surface) > best[0]:
                best = (len(surface), entity_id, surface)
        if best is None:
            return None
        return (best[1], best[2])

    # -- EntityRuler integration -------------------------------------------

    def ruler_patterns(self) -> tuple[RulerPattern, ...]:
        """Patterns for the spaCy EntityRuler: named aliases only.

        Anaphors are excluded by design (module docstring). Patterns are emitted
        longest-first so the phrase matcher prefers ``Project Aurora`` over ``Aurora``.
        """
        patterns: list[RulerPattern] = []
        for entity in self._entities:
            surfaces = {entity.label, *entity.aliases}
            for surface in surfaces:
                if not surface or surface.lower().startswith("the "):
                    continue
                patterns.append(
                    RulerPattern(
                        surface=surface,
                        entity_type=entity.type.value,
                        canonical_label=entity.label,
                        entity_id=entity.entity_id,
                    )
                )
        patterns.sort(key=lambda p: (-len(p.surface), p.surface))
        return tuple(patterns)


class EntityResolver:
    """Maps mention spans to canonical entities (specification section 7).

    Invariants: a resolved mention always carries both the canonical id and the
    original surface text, so the UI can always show the transformation.
    """

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    def resolve(self, mention: EntityMention) -> EntityMention:
        """Return a copy of ``mention`` annotated with its canonical entity."""
        entity_id = self._registry.resolve_surface(mention.text)
        if entity_id is None:
            return mention
        entity = self._registry.entity(entity_id)
        if entity is None:
            return mention
        from dataclasses import replace

        return replace(
            mention,
            entity_id=entity_id,
            canonical_label=entity.label,
            entity_type=entity.type,
        )

    def resolve_all(self, mentions: Iterable[EntityMention]) -> tuple[EntityMention, ...]:
        return tuple(self.resolve(m) for m in mentions)

    def resolve_text(self, text: str) -> tuple[str, str] | None:
        """Convenience wrapper used by the query parser and by rules needing a label."""
        return self._registry.resolve_surface(text), text

    def find_in_text(self, text: str) -> tuple[str, str] | None:
        return self._registry.find_in_text(text)


def build_entity_id_for(label: str, entity_type: EntityType) -> str:
    """Convenience re-export so rules need not import :mod:`src.ids` directly."""
    return make_entity_id(entity_type, label) or slugify(label)


__all__ = [
    "EntityRegistry",
    "EntityResolver",
    "normalise_surface",
    "is_ambiguous_pronoun",
    "PRONOUN_SURFACES",
    "RulerPattern",
]
