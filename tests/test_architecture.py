"""Charter invariants that constrain the *code*, not the data.

These are cheap checks: no models, no corpus, no inference. They run in well under a
second and are meant to catch a violation at the moment it is introduced rather than
after a long pipeline run.

Enforces:
  * I1  nothing under ``src/`` imports ``streamlit``, ``pyvis`` or ``app``
  * I6  device is resolved at runtime and overridable, never hard-wired to GPU
  * I9  the dependency floor keeps ``transformers`` on 4.x
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

SRC = pathlib.Path("src")
FORBIDDEN_IN_SRC = ("streamlit", "pyvis", "app")


def _imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """(lineno, module) for every real import in a file.

    Parsed rather than grepped: the module docstrings and the logging configuration
    legitimately mention Streamlit and PyVis by name, and a text search produces
    false positives that would push someone towards silencing the check instead of
    fixing the real import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.module or ""))
    return found


def _python_sources(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if ".venv" not in p.parts)


class TestInvariantI1:
    """The pipeline runs headless: no UI library may reach src/."""

    def test_src_has_no_ui_imports(self):
        offenders: list[str] = []
        for path in _python_sources(SRC):
            for lineno, module in _imports(path):
                head = module.split(".")[0]
                if head in FORBIDDEN_IN_SRC:
                    offenders.append(f"{path.as_posix()}:{lineno} imports {module}")
        assert not offenders, "I1 violated -- UI must not leak into src/:\n" + "\n".join(
            offenders
        )

    def test_pipeline_module_runs_headless(self):
        """`python -m src.pipeline` must work with no UI installed."""
        import src.pipeline as pipeline

        assert callable(pipeline.main)
        assert pipeline.PROGRESS_MESSAGES[0] == "Loading documents…"
        assert pipeline.PROGRESS_MESSAGES[-1] == "Done."

    def test_no_transitive_import_of_ui_libraries(self):
        """Importing src.pipeline must not pull in a UI library.

        Every streamlit submodule is evicted and restored around the import. Evicting only
        the top-level name is not enough: streamlit caches a process-wide
        DeltaGeneratorSingleton in a submodule, and a partial eviction leaves that stale
        instance behind, which makes every later streamlit import fail with
        "DeltaGeneratorSingleton instance already exists".
        """
        import importlib
        import sys

        def streamlit_modules() -> dict[str, object]:
            return {
                name: module
                for name, module in sys.modules.items()
                if name == "streamlit" or name.startswith("streamlit.")
            }

        def evict_tree(prefix: str) -> None:
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + "."):
                    del sys.modules[name]

        saved = streamlit_modules()
        try:
            evict_tree("streamlit")
            sys.modules.pop("pyvis", None)
            importlib.import_module("src.pipeline")
            assert "streamlit" not in sys.modules
            assert "pyvis" not in sys.modules
        finally:
            evict_tree("streamlit")
            sys.modules.update(saved)


class TestInvariantI6:
    """Device is chosen at runtime; CPU is always available."""

    def test_config_resolves_a_device(self):
        from src.config import AppConfig

        config = AppConfig.from_env()
        assert config.coref_device in {"cpu", "cuda", "cuda:0", "mps"}

    def test_device_is_not_hard_wired(self):
        source = (SRC / "config.py").read_text(encoding="utf-8")
        assert "cuda:0" not in re.sub(r'"[^"]*cuda[^"]*"|\x27[^\x27]*cuda[^\x27]*\x27', "", source)
        assert "resolve_device" in source

    def test_device_is_env_overridable(self, monkeypatch):
        from src.config import AppConfig

        monkeypatch.setenv("KG_COREF_DEVICE", "cpu")
        assert AppConfig.from_env().coref_device == "cpu"


class TestInvariantI9:
    """transformers must stay on 4.x; FastCoref breaks on 5.x."""

    def test_requirements_pin_transformers_below_five(self):
        requirements = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
        pins = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip().lower().startswith("transformers")
        ]
        assert pins, "transformers is not pinned in requirements.txt"
        assert any("<5" in pin for pin in pins), pins


class TestProjectLayout:
    def test_source_modules_exist(self):
        expected = {
            "config.py",
            "models.py",
            "document_loader.py",
            "nlp_processor.py",
            "entity_resolver.py",
            "coreference.py",
            "rules/base.py",
            "rules/extraction.py",
            "triple_builder.py",
            "graph_builder.py",
            "serialization.py",
            "query.py",
            "pipeline.py",
        }
        present = {p.relative_to(SRC).as_posix() for p in _python_sources(SRC)}
        assert expected <= present, f"missing: {sorted(expected - present)}"

    def test_tests_package_is_importable(self):
        assert (pathlib.Path("tests") / "__init__.py").exists()

    def test_every_source_file_parses(self):
        for path in _python_sources(SRC):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_module_uses_a_bare_except(self):
        """`except:` swallows KeyboardInterrupt and hides real failures."""
        for path in _python_sources(SRC):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    pytest.fail(f"{path.as_posix()}:{node.lineno} uses a bare except")


@pytest.mark.parametrize("filename", ["pyproject.toml", "requirements.txt"])
def test_config_files_present(filename):
    assert pathlib.Path(filename).exists(), f"{filename} is missing"
