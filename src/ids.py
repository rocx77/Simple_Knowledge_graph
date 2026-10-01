"""Stable identifier construction.

Ids appear in artifacts, tests and the UI explainability trace, so their format is a
public contract (rule N2). Changing it is a breaking change to the project report.

Format::

    person:arun_mehta
    product:orionedge
    project:project_aurora
    triple:person:arun_mehta--works_on->product:orionedge@doc_02_team
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import EntityType

_NON_WORD = re.compile(r"[^a-z0-9]+")
_HONORIFICS = frozenset({"dr", "mr", "mrs", "ms", "prof", "sir", "madam"})
_POSSESSIVE = re.compile(r"'s$", re.IGNORECASE)


def slugify(text: str) -> str:
    """Normalise a surface form into a lower snake-case slug.

    Strips a trailing possessive, then any leading honorific, then collapses every
    run of non-alphanumeric characters into a single underscore.

    >>> slugify("Dr. Mira Sen")
    'mira_sen'
    >>> slugify("Aurora's")
    'aurora'
    >>> slugify("Entity Resolution Engine")
    'entity_resolution_engine'
    """
    cleaned = _POSSESSIVE.sub("", text.strip())
    slug = _NON_WORD.sub("_", cleaned.lower()).strip("_")
    parts = [p for p in slug.split("_") if p]
    while parts and parts[0] in _HONORIFICS:
        parts.pop(0)
    return "_".join(parts)


def make_entity_id(entity_type: "EntityType | str", label: str) -> str:
    """Build a canonical entity id, e.g. ``("PERSON", "Dr. Mira Sen") -> "person:mira_sen"``.

    ``LITERAL`` ids are prefixed ``literal:`` to keep them clearly distinguishable from
    named entities in artifacts and in the UI.
    """
    type_value = getattr(entity_type, "value", entity_type)
    prefix = "literal" if str(type_value).upper() == "LITERAL" else str(type_value).lower()
    return f"{prefix}:{slugify(label)}"


def make_triple_id(subject_id: str, relation: str, object_id: str, document_id: str) -> str:
    """Build a deterministic triple id.

    Deterministic matters: it lets a graph rebuilt from the same documents produce
    byte-identical artifacts, which is what makes "nothing changed" verifiable.
    """
    relation_value = getattr(relation, "value", relation)
    return f"triple:{subject_id}--{relation_value}->{object_id}@{document_id}"


def make_mention_id(document_id: str, start_char: int, end_char: int, surface: str) -> str:
    """Build a stable id for a single entity mention."""
    return f"mention:{document_id}:{start_char}:{end_char}:{slugify(surface)}"


__all__ = ["slugify", "make_entity_id", "make_triple_id", "make_mention_id"]