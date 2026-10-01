"""Application configuration.

Every tunable lives here (rule M4: all domain/environment knowledge in one place).
Nothing else in the codebase reads ``os.environ``.

Environment variables
---------------------
``KG_DOCS_DIR``                  corpus directory          default ``data/docs``
``KG_ARTIFACTS_DIR``             artifact output directory default ``artifacts``
``KG_SPACY_MODEL``               spaCy pipeline            default ``en_core_web_sm``
``COREF_MODEL_NAME_OR_PATH``     FastCoref checkpoint      default ``biu-nlp/f-coref``
``COREF_DEVICE``                 force a device            default: auto-detect
``KG_ENABLE_OPTIONAL_RELATIONS`` ``true``/``false``        default ``true``
``KG_LOG_LEVEL``                 logging level             default ``INFO``
``KG_MAX_PATH_LENGTH``           BFS path cap for queries  default ``3``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigurationError

TRUTHY = frozenset({"1", "true", "yes", "on"})
FALSY = frozenset({"0", "false", "no", "off"})

DEFAULT_MAX_PATH_LENGTH = 3


def _project_root() -> Path:
    """Repository root: the parent of the ``src`` package directory."""
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    return candidate.resolve()


def resolve_device(explicit: str | None = None) -> str:
    """Resolve the FastCoref device.

    Precedence (specification section 1):

    1. an explicit request (``COREF_DEVICE`` or the ``explicit`` argument),
    2. ``cuda:0`` when CUDA is actually available,
    3. ``cpu``.

    A request for CUDA on a machine without CUDA **falls back to CPU** and logs a
    warning rather than raising. GPU-only execution is forbidden (charter invariant
    I6), so degradation here is always the correct behaviour.

    ``explicit`` is accepted so tests can exercise the matrix without mutating the
    process environment.
    """
    requested = explicit if explicit is not None else os.environ.get("COREF_DEVICE")

    cuda_available = False
    torch_version = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch is optional for graph/query-only use
        logging.getLogger(__name__).debug("torch not importable; assuming CPU execution")

    if requested:
        normalized = requested.strip().lower()
        if normalized.startswith("cuda") and not cuda_available:
            logging.getLogger(__name__).warning(
                "COREF_DEVICE=%s requested but CUDA is unavailable (torch %s); falling back to CPU",
                requested,
                torch_version,
            )
            return "cpu"
        if normalized in {"cpu", "cuda", "cuda:0"}:
            return "cpu" if normalized == "cpu" else "cuda:0"
        logging.getLogger(__name__).warning(
            "COREF_DEVICE=%s is not a recognised device; falling back to auto-detection", requested
        )

    return "cuda:0" if cuda_available else "cpu"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable application settings.

    Frozen so a stage cannot mutate shared configuration mid-run (rule O4).
    """

    root: Path
    docs_dir: Path
    artifacts_dir: Path
    spacy_model: str = "en_core_web_sm"
    coref_model: str = "biu-nlp/f-coref"
    coref_device: str = field(default_factory=resolve_device)
    enable_optional_relations: bool = True
    log_level: str = "INFO"
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build a configuration from environment variables, validating as we go."""
        root = _project_root()
        max_path_raw = os.environ.get("KG_MAX_PATH_LENGTH")
        try:
            max_path_length = int(max_path_raw) if max_path_raw else DEFAULT_MAX_PATH_LENGTH
        except ValueError as exc:
            raise ConfigurationError(
                f"KG_MAX_PATH_LENGTH must be an integer, got {max_path_raw!r}"
            ) from exc
        if max_path_length < 1:
            raise ConfigurationError("KG_MAX_PATH_LENGTH must be >= 1")

        optional_raw = os.environ.get("KG_ENABLE_OPTIONAL_RELATIONS", "true").strip().lower()
        if optional_raw in TRUTHY:
            enable_optional = True
        elif optional_raw in FALSY:
            enable_optional = False
        else:
            raise ConfigurationError(
                f"KG_ENABLE_OPTIONAL_RELATIONS must be a boolean, got {optional_raw!r}"
            )

        return cls(
            root=root,
            docs_dir=_env_path("KG_DOCS_DIR", root / "data" / "docs"),
            artifacts_dir=_env_path("KG_ARTIFACTS_DIR", root / "artifacts"),
            spacy_model=os.environ.get("KG_SPACY_MODEL", "en_core_web_sm"),
            coref_model=os.environ.get("COREF_MODEL_NAME_OR_PATH", "biu-nlp/f-coref"),
            coref_device=resolve_device(),
            enable_optional_relations=enable_optional,
            log_level=os.environ.get("KG_LOG_LEVEL", "INFO").upper(),
            max_path_length=max_path_length,
        )

    # -- derived locations -------------------------------------------------

    @property
    def graph_json(self) -> Path:
        return self.artifacts_dir / "graph.json"

    @property
    def triples_csv(self) -> Path:
        return self.artifacts_dir / "triples.csv"

    @property
    def entities_json(self) -> Path:
        return self.artifacts_dir / "entities.json"

    @property
    def pipeline_report_json(self) -> Path:
        return self.artifacts_dir / "pipeline_report.json"

    def ensure_artifacts_dir(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, str]:
        """Flat dict for the pipeline report."""
        return {
            "spacy_model": self.spacy_model,
            "coref_model": self.coref_model,
            "coref_device": self.coref_device,
            "docs_dir": str(self.docs_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "enable_optional_relations": str(self.enable_optional_relations).lower(),
            "max_path_length": str(self.max_path_length),
        }


__all__ = ["AppConfig", "resolve_device", "DEFAULT_MAX_PATH_LENGTH"]
