"""S8 -- run every stage in order and report progress.

``python -m src.pipeline`` regenerates all four artifacts and prints a summary.

Two constraints shape this module:

* No Streamlit import anywhere in ``src/`` (specification S8 acceptance). The UI calls
  :meth:`KnowledgeGraphPipeline.run` with its own ``progress_cb``; the pipeline knows
  nothing about Streamlit.
* One spaCy ``Language`` is loaded once and shared. S3 coreference reuses the S2 object,
  so reloading per stage would silently double the largest cost in the run.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .coreference import (
    CoreferenceResolver,
    DeterministicCorefFallback,
    FastCorefResolver,
)
from .document_loader import DocumentLoader
from .entity_resolver import EntityRegistry
from .graph_builder import GraphBuilder
from .logging_utils import configure_logging
from .models import MentionResolution, Triple
from .nlp_processor import NLPProcessor
from .rules import RelationExtractor
from .serialization import ArtifactPaths, ArtifactWriter, PipelineReport, relation_histogram
from .triple_builder import TripleBuilder

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

#: Progress messages required by specification section 32, in order.
PROGRESS_MESSAGES = (
    "Loading documents…",
    "Running spaCy…",
    "Resolving coreferences…",
    "Extracting relations…",
    "Building graph…",
    "Saving artifacts…",
    "Done.",
)


@dataclass(slots=True)
class StageReport:
    """Outcome of one stage."""

    name: str
    elapsed_seconds: float = 0.0
    ok: bool = True
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "ok": self.ok,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass(slots=True)
class PipelineResult:
    """Everything a caller might want after a run, UI or otherwise."""

    graph: Any
    triples: tuple[Triple, ...]
    resolutions: tuple[MentionResolution, ...]
    artifacts: ArtifactPaths
    report: PipelineReport

    @property
    def summary(self) -> str:
        return (
            f"{self.report.counts.get('nodes', 0)} nodes, "
            f"{self.report.counts.get('triples', 0)} edges"
        )


class KnowledgeGraphPipeline:
    """Runs S1 to S7 and serialises the result."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.from_env()

    def run(self, progress_cb: ProgressCallback | None = None) -> PipelineResult:
        report = PipelineReport(
            device=self._config.coref_device,
            environment={**self._config.summary(), "python": platform.python_version()},
            versions=_library_versions(),
        )

        stages: list[StageReport] = []
        say = progress_cb or (lambda _message: None)

        say(PROGRESS_MESSAGES[0])
        registry = EntityRegistry.load()
        documents = DocumentLoader(self._config).load()
        stages.append(_timed("load_documents", lambda: None))

        say(PROGRESS_MESSAGES[1])
        processor = NLPProcessor(self._config, registry.ruler_patterns())
        # Loaded once here and reused by S3; the assignment is the point of the stage.
        stages.append(_timed("load_spacy", processor.load))

        processed = []
        for document in documents:
            processed.append(processor.process(document))
        stages.append(_timed("process_documents", lambda: None))

        say(PROGRESS_MESSAGES[2])
        coref = CoreferenceResolver(
            self._config,
            registry,
            FastCorefResolver(self._config, processor.load()),
            DeterministicCorefFallback(registry),
        )
        pairs: list[tuple[Any, dict[tuple[int, int], MentionResolution]]] = []
        resolutions: list[MentionResolution] = []
        for item in processed:
            item_resolutions = coref.resolve(item)
            resolutions.extend(item_resolutions)
            pairs.append(
                (item, {(r.start_char, r.end_char): r for r in item_resolutions})
            )
        stages.append(_timed("resolve_coreference", lambda: None))

        say(PROGRESS_MESSAGES[3])
        extractor = RelationExtractor(
            enable_optional_relations=self._config.enable_optional_relations
        )
        candidates, skipped = extractor.extract(pairs, registry=registry)
        triples, rejections = TripleBuilder(registry).build(candidates)
        stages.append(_timed("extract_and_build_triples", lambda: None))

        say(PROGRESS_MESSAGES[4])
        graph = GraphBuilder(registry).build(triples, resolutions)
        stages.append(_timed("build_graph", lambda: None))

        say(PROGRESS_MESSAGES[5])
        self._fill_report(
            report,
            triples=triples,
            resolutions=resolutions,
            skipped=skipped,
            rejections=rejections,
            graph=graph,
            documents=len(documents),
        )
        artifacts = ArtifactWriter(self._config, registry).write_all(
            graph, triples, resolutions, report
        )
        stages.append(_timed("write_artifacts", lambda: None))

        report.stages = [stage.to_dict() for stage in stages]
        say(PROGRESS_MESSAGES[6])

        return PipelineResult(
            graph=graph,
            triples=triples,
            resolutions=tuple(resolutions),
            artifacts=artifacts,
            report=report,
        )

    def _fill_report(
        self,
        report: PipelineReport,
        *,
        triples: tuple[Triple, ...],
        resolutions: list[MentionResolution],
        skipped: tuple,
        rejections: tuple,
        graph: Any,
        documents: int,
    ) -> None:
        report.counts = {
            "documents": documents,
            "mentions": len(resolutions),
            "resolved_mentions": sum(1 for r in resolutions if r.entity_id),
            "unresolved_mentions": sum(1 for r in resolutions if not r.entity_id),
            "relation_candidates": len(triples),
            "triples": len(triples),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "skipped_candidates": len(skipped),
        }
        report.counts.update(
            {f"rejected_{k}": v for k, v in rejections._asdict().items()}
        )
        report.relation_histogram = relation_histogram(triples)
        report.unresolved_mentions = [
            {
                "document_id": r.document_id,
                "sentence_index": r.sentence_index,
                "text": r.mention,
                "detail": r.conflict_detail,
            }
            for r in resolutions
            if not r.entity_id
        ]
        report.coref_conflicts = [
            {
                "document_id": r.document_id,
                "text": r.mention,
                "resolved_to": r.resolved_label,
                "detail": r.conflict_detail,
            }
            for r in resolutions
            if r.conflict
        ]
        report.skipped_candidates = [
            {
                "rule_id": s.rule_id,
                "relation": s.relation,
                "document_id": s.document_id,
                "sentence_index": s.sentence_index,
                "reason": s.reason,
            }
            for s in skipped
        ]


# -- helpers ----------------------------------------------------------------


def _timed(name: str, action: Callable[[], Any]) -> StageReport:
    """Run ``action``, recording how long it took and whether it raised.

    A stage failure is captured rather than propagated so the report still names the
    stage that broke; ``run`` re-raises afterwards.
    """
    started = time.perf_counter()
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by the caller
        logger.exception("stage %s failed", name)
        return StageReport(
            name=name, ok=False, error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
    return StageReport(name=name, elapsed_seconds=time.perf_counter() - started)


def _library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour the output depends on."""
    from importlib.metadata import PackageNotFoundError, version

    names = {
        "spacy": "spacy",
        "fastcoref": "fastcoref",
        "networkx": "networkx",
        "pyvis": "pyvis",
        "transformers": "transformers",
        "torch": "torch",
    }
    out: dict[str, str] = {}
    for key, dist in names.items():
        try:
            out[key] = version(dist)
        except PackageNotFoundError:  # pragma: no cover - optional dependency
            out[key] = "not installed"
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.pipeline``.

    ``argv`` is part of the signature a future CLI would use, but nothing is parsed yet:
    configuration comes entirely from the environment, per the charter.
    """
    del argv
    configure_logging("INFO")
    config = AppConfig.from_env()

    def progress(message: str) -> None:
        print(message, flush=True)

    started = time.perf_counter()
    result = KnowledgeGraphPipeline(config).run(progress_cb=progress)
    elapsed = time.perf_counter() - started

    print()
    print(f"Entities      {result.report.counts['nodes']}")
    print(f"Relations     {result.report.counts['edges']}")
    print(f"Mentions      {result.report.counts['mentions']} "
          f"({result.report.counts['unresolved_mentions']} unresolved)")
    print()
    print("Artifacts")
    for name, path in result.artifacts.as_dict().items():
        print(f"  {name:<9} {path}")
    print()
    print("Relation histogram")
    for relation, count in result.report.relation_histogram.items():
        print(f"  {relation:<18} {count}")
    if result.report.warnings:
        print()
        print("Warnings")
        for warning in result.report.warnings:
            print(f"  {warning}")
    print()
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
