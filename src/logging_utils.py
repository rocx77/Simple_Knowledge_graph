"""Logging configuration.

Rule O7: model-loading libraries flood stdout at INFO level (spaCy, torch,
transformers, FastCoref and its HuggingFace client between them emit dozens of
"HTTP Request" lines per run). The pipeline silences those specific loggers at the
boundary and leaves its own loggers untouched, so the pipeline's own progress output
stays readable both in the terminal and inside the Streamlit UI.
"""

from __future__ import annotations

import logging
import os

#: Third-party loggers that must never pollute pipeline output.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "datasets",
    "datasets.utils",
    "filelock",
    "fastcoref",
    "sentence_transformers",
    "absl",
    "numba",
    "matplotlib",
    "streamlit",
)

LOG_FORMAT = "%(levelname)-8s %(name)-28s %(message)s"


def configure_logging(level: str | int | None = None, *, force: bool = True) -> None:
    """Configure root logging and silence noisy third-party loggers.

    Args:
        level: level name or numeric level. Defaults to ``KG_LOG_LEVEL`` then ``INFO``.
        force: remove existing handlers first. Useful in tests.
    """
    resolved = level or os.environ.get("KG_LOG_LEVEL", "INFO")
    numeric = resolved if isinstance(resolved, int) else logging.getLevelName(str(resolved).upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    root.setLevel(numeric)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Convention: ``logging.getLogger(__name__)``."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "NOISY_LOGGERS", "LOG_FORMAT"]
