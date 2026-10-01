"""Visual language for the explorer.

Kept free of Streamlit imports so the palette can be unit-tested directly.
"""

from __future__ import annotations

ENTITY_COLORS: dict[str, str] = {
    "PERSON": "#4C8DF6",
    "ORGANIZATION": "#F2994A",
    "PRODUCT": "#9B51E0",
    "PROJECT": "#27AE60",
    "COMPONENT": "#EB5757",
    "LITERAL": "#828282",
}

FALLBACK_COLOR = "#B0B0B0"

# Edge styling carries meaning, not decoration: a lighter dashed line marks a
# relation the pipeline inferred rather than one the text stated outright.
RELATION_COLOR = "#5C6B7A"
INFERRED_RELATION_COLOR = "#A0A0A0"

PAGE_CSS = """
<style>
  .kg-subtitle    { font-size: 0.95rem; opacity: 0.7; margin-top: -0.6rem; }

  .kg-kpi          { border-radius: 12px; padding: 0.9rem 1.1rem; height: 100%;
                     border: 1px solid rgba(128,128,128,0.25); }
  .kg-kpi-value    { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
  .kg-kpi-label    { font-size: 0.72rem; text-transform: uppercase;
                     letter-spacing: 0.09em; opacity: 0.65; }

  .kg-card         { border-radius: 12px; padding: 1rem 1.15rem;
                     border: 1px solid rgba(128,128,128,0.25); margin-bottom: 0.8rem; }
  .kg-card-title   { font-size: 0.74rem; text-transform: uppercase;
                     letter-spacing: 0.09em; opacity: 0.6; margin-bottom: 0.35rem; }
  .kg-answer       { font-size: 1.15rem; font-weight: 600; }

  .kg-evidence-quote { font-style: italic; opacity: 0.9;
                       border-left: 3px solid rgba(128,128,128,0.4);
                       padding-left: 0.7rem; margin: 0.35rem 0; }
  .kg-resolved    { font-size: 0.85rem; padding: 0.25rem 0.55rem; display: inline-block;
                    border-radius: 6px; background: rgba(128,128,128,0.16); }

  .kg-section     { font-size: 1.05rem; font-weight: 650; margin: 1.4rem 0 0.5rem; }
  .kg-muted       { font-size: 0.82rem; opacity: 0.65; }
  .kg-swatch      { display: inline-block; width: 11px; height: 11px; border-radius: 50%%;
                    margin-right: 6px; vertical-align: middle; }
</style>
"""


def color_for(entity_type: str) -> str:
    """Colour for an entity type, tolerating an unknown type rather than raising."""
    return ENTITY_COLORS.get(str(entity_type).upper(), FALLBACK_COLOR)


def swatch(entity_type: str) -> str:
    """HTML legend swatch for an entity type."""
    return f'<span class="kg-swatch" style="background:{color_for(entity_type)}"></span>'


def swatch_legend(entity_types: list[str]) -> str:
    """One-line HTML legend, ordered by the canonical palette order."""
    ordered = [t for t in ENTITY_COLORS if t in entity_types]
    ordered += sorted(t for t in entity_types if t not in ENTITY_COLORS)
    return " &nbsp; ".join(f"{swatch(t)} {t}" for t in ordered)
