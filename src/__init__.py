"""Knowledge Graph MVP -- offline NLP pipeline to queryable knowledge graph.

Layering rule (charter invariant I1): nothing in this package may import ``streamlit``
or ``pyvis``. Run the whole pipeline headless with::

    python -m src.pipeline

Sub-modules are grouped by pipeline stage; see ``docs/02_PIPELINE_PLAN.md``.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
