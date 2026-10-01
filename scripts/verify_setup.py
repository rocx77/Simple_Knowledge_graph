"""Environment doctor for the Knowledge Graph MVP.

Verifies that every runtime dependency the pipeline needs is importable, that the
spaCy English model is present, and that the coreference model can actually run
inference on this machine. Run it before anything else:

    .venv\\Scripts\\python.exe scripts\\verify_setup.py
    .venv\\Scripts\\python.exe scripts\\verify_setup.py --warm-cache

Exit code 0 means the environment is ready. Any non-zero exit lists what broke.

This script deliberately imports nothing from ``src`` so that it can diagnose a
broken environment before the pipeline is installed into the path.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPACY_MODEL = os.environ.get("KG_SPACY_MODEL", "en_core_web_sm")
COREF_MODEL = os.environ.get("COREF_MODEL_NAME_OR_PATH", "biu-nlp/f-coref")
SAMPLE_TEXT = (
    "Arun Mehta joined Aether Analytics. He works on OrionEdge. "
    "The platform is deployed at Helios Bank."
)

OK = "PASS"
FAIL = "FAIL"
WARN = "WARN"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) + 2
        icon = {OK: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]"}
        lines = [f"{icon[c.status]} {c.name.ljust(width)} {c.detail}".rstrip() for c in self.checks]
        lines.append("")
        lines.append(f"total={len(self.checks)}  passed={len(self.checks) - len(self.failures)}"
                     f"  warnings={len(self.warnings)}  failed={len(self.failures)}")
        return "\n".join(lines)


def resolve_device() -> str:
    """Resolve the FastCoref device.

    Precedence: explicit ``COREF_DEVICE`` env var, then CUDA if available, else CPU.
    An unusable explicit value falls back to CPU with a warning rather than crashing.
    """
    override = os.environ.get("COREF_DEVICE")
    if override:
        try:
            import torch

            if override.startswith("cuda") and not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
        return override
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def check_python(report: Report) -> None:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info >= (3, 10)
    report.add("python interpreter", OK if ok else FAIL, f"{version} ({sys.executable})")
    if not ok:
        report.add("python version floor", FAIL, "3.10+ required")


def check_imports(report: Report) -> None:
    modules = {
        "spacy": "spacy",
        "networkx": "networkx",
        "pyvis": "pyvis",
        "streamlit": "streamlit",
        "pandas": "pandas",
        "torch": "torch",
        "fastcoref": "fastcoref",
    }
    for label, module in modules.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            report.add(f"import {label}", OK, version)
        except Exception as exc:  # noqa: BLE001 - doctor must report, not crash
            report.add(f"import {label}", FAIL, f"{type(exc).__name__}: {exc}")


def check_spacy_model(report: Report) -> None:
    try:
        import spacy
    except Exception as exc:  # noqa: BLE001
        report.add("spacy model", FAIL, f"spacy unavailable: {exc}")
        return

    try:
        nlp = spacy.load(SPACY_MODEL)
    except Exception as exc:  # noqa: BLE001
        report.add("spacy model", FAIL,
                   f"could not load {SPACY_MODEL!r}: {exc}\n"
                   f"           install it with:\n"
                   f"           .venv\\Scripts\\python.exe -m pip install "
                   f"https://github.com/explosion/spacy-models/releases/download/"
                   f"{SPACY_MODEL}-3.8.0/{SPACY_MODEL}-3.8.0-py3-none-any.whl")
        return

    required = {"tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"}
    missing = required.difference(nlp.pipe_names)
    report.add("spacy model", OK if not missing else FAIL,
               f"{SPACY_MODEL} | pipes={nlp.pipe_names}"
               + (f" MISSING={sorted(missing)}" if missing else ""))

    doc = nlp("Aether Analytics was founded by Dr. Mira Sen.")
    ents = [(e.text, e.label_) for e in doc.ents]
    report.add("spacy inference", OK, f"tokens={len(doc)} entities={ents}")
    report.add("spacy NER reliability note", WARN,
               "generic NER may mislabel fictional names (e.g. 'Aether Analytics'); "
               "the EntityRuler layer is the corrective mechanism by design")


def check_device(report: Report) -> None:
    device = resolve_device()
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"resolved={device} torch={torch.__version__} cuda_available={cuda}"
        if cuda:
            detail += f" gpu={torch.cuda.get_device_name(0)}"
        report.add("torch device", OK, detail)
    except Exception as exc:  # noqa: BLE001
        report.add("torch device", FAIL, f"{type(exc).__name__}: {exc}")


def check_hf_cache(report: Report) -> None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        report.add("fastcoref weights cache", WARN, "huggingface_hub not inspectable")
        return

    hit = try_to_load_from_cache(COREF_MODEL, "pytorch_model.bin")
    if isinstance(hit, str):
        size_mb = Path(hit).stat().st_size / 1_048_576
        report.add("fastcoref weights cache", OK, f"cached at {hit} ({size_mb:.1f} MB)")
    else:
        report.add("fastcoref weights cache", WARN,
                   f"{COREF_MODEL} not cached yet; first run will download ~362 MB "
                   f"(use --warm-cache to do it now)")


def check_coref(report: Report, warm_cache: bool) -> Any:
    """Load FastCoref and run one prediction. Returns the loaded model or None."""
    device = resolve_device()
    try:
        import spacy
        from fastcoref import FCoref

        nlp = spacy.load(SPACY_MODEL)
        started = time.perf_counter()
        model = FCoref(model_name_or_path=COREF_MODEL, device=device, nlp=nlp,
                       enable_progress_bar=False)
        load_s = time.perf_counter() - started
        report.add("fastcoref model load", OK,
                   f"{COREF_MODEL} on {device} in {load_s:.1f}s")
    except Exception as exc:  # noqa: BLE001
        report.add("fastcoref model load", FAIL, f"{type(exc).__name__}: {exc}")
        return None

    try:
        started = time.perf_counter()
        preds = model.predict(texts=[SAMPLE_TEXT])
        clusters = preds[0].get_clusters()
        infer_s = time.perf_counter() - started
        report.add("fastcoref inference", OK,
                   f"{len(clusters)} cluster(s) in {infer_s:.1f}s -> {clusters}")
    except Exception as exc:  # noqa: BLE001
        report.add("fastcoref inference", FAIL, f"{type(exc).__name__}: {exc}")
        return model

    if warm_cache:
        report.add("fastcoref cache warm", OK, "weights already present after inference")

    return model


def check_repo_layout(report: Report, root: Path) -> None:
    docs_dir = root / "data" / "docs"
    if docs_dir.is_dir():
        found = sorted(p.name for p in docs_dir.glob("*.txt"))
        if found:
            report.add("corpus", OK, f"{len(found)} document(s): {', '.join(found)}")
        else:
            report.add("corpus", WARN, "data/docs exists but contains no .txt files")
    else:
        report.add("corpus", WARN, "data/docs not created yet (Phase 2)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Knowledge Graph MVP environment.")
    parser.add_argument("--warm-cache", action="store_true",
                        help="force a FastCoref inference pass so the HF weights land in the local cache")
    parser.add_argument("--skip-coref", action="store_true",
                        help="skip loading the coreference model (fast structural check only)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (defaults to the parent of scripts/)")
    args = parser.parse_args()

    report = Report()

    print("Knowledge Graph MVP - environment doctor")
    print("=" * 78)
    print()

    check_python(report)
    check_imports(report)
    check_spacy_model(report)
    check_device(report)
    check_hf_cache(report)
    if not args.skip_coref:
        check_coref(report, warm_cache=args.warm_cache)
    check_repo_layout(report, args.root)

    print(report.render())
    print()

    if report.failures:
        print("RESULT: FAILED - fix the items marked FAIL before continuing.")
        return 1
    if report.warnings:
        print("RESULT: READY (with warnings - review the WARN lines above)")
        return 0
    print("RESULT: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())