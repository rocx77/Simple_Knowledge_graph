"""Corpus integrity checker.

The demonstration documents are fixed by the master specification. If their
text ever drifts, every downstream expectation (triples, queries, tests) becomes
untrustworthy, so this script makes that drift loud and immediate.

    .venv\\Scripts\\python.exe scripts\\verify_corpus.py
    .venv\\Scripts\\python.exe scripts\\verify_corpus.py --write-manifest

Exit code 0 means the corpus matches the specification byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.json"

#: Documents required by the master specification, in load order.
EXPECTED_FILES = [
    "doc_01_company.txt",
    "doc_02_team.txt",
    "doc_03_deployment.txt",
    "doc_04_aurora.txt",
    "doc_05_customer.txt",
    "doc_06_graphmatch.txt",
    "doc_07_ledgerline.txt",
    "doc_08_auditlens.txt",
    "doc_09_working_group.txt",
    "doc_10_openai.txt",
    "doc_11_deepmind.txt",
    "doc_12_linux.txt",
    "doc_13_cern.txt",
    "doc_14_python.txt",
    "doc_15_mozilla.txt",
    "doc_16_bridge.txt",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for name in EXPECTED_FILES:
        path = DOCS_DIR / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        manifest[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "characters": len(raw.decode("utf-8")),
            "sentences_estimate": raw.decode("utf-8").count(".") - 1,
        }
    return manifest


def verify(manifest: dict[str, dict[str, object]]) -> list[str]:
    """Return a list of human-readable problems. Empty list means the corpus is intact."""
    problems: list[str] = []

    present = sorted(p.name for p in DOCS_DIR.glob("*.txt")) if DOCS_DIR.is_dir() else []
    if not DOCS_DIR.is_dir():
        return [f"missing directory: {DOCS_DIR}"]

    for name in EXPECTED_FILES:
        if name not in present:
            problems.append(f"MISSING  {name} (required by specification)")

    for name in present:
        if name not in EXPECTED_FILES:
            problems.append(f"EXTRA    {name} (unexpected file in data/docs)")

    for name, expected in manifest.items():
        path = DOCS_DIR / name
        if not path.is_file():
            problems.append(f"MISSING  {name} (present in manifest, absent on disk)")
            continue
        actual = sha256_of(path)
        if actual != expected["sha256"]:
            problems.append(
                f"DRIFTED  {name}\n"
                f"           expected sha256 {expected['sha256']}\n"
                f"           actual   sha256 {actual}\n"
                f"           the corpus text no longer matches the specification"
            )

    for name in present:
        path = DOCS_DIR / name
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            problems.append(f"EMPTY    {name} (contains no non-whitespace characters)")
        if "\n" in text.strip():
            problems.append(f"MULTILINE {name} (specification defines these as single paragraphs)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the demonstration corpus.")
    parser.add_argument("--write-manifest", action="store_true",
                        help="(re)generate data/corpus_manifest.json from the current files")
    args = parser.parse_args()

    if args.write_manifest or not MANIFEST_PATH.is_file():
        manifest = build_manifest()
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST_PATH.relative_to(ROOT)} with {len(manifest)} entries")
        if not args.write_manifest:
            print("(manifest did not exist, so it was created rather than verified)")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    problems = verify(manifest)

    print("Knowledge Graph MVP - corpus integrity")
    print("=" * 78)
    for name, meta in manifest.items():
        print(f"  {name:<26} {meta['bytes']:>4} B  {meta['characters']:>4} chars  sha256={meta['sha256'][:16]}...")
    print()

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        print()
        print(f"RESULT: FAILED - {len(problems)} problem(s). The corpus must match the spec exactly.")
        return 1

    print(f"[ OK ] all {len(manifest)} documents present, non-empty, single-paragraph, hash-verified")
    print()
    print("RESULT: INTACT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())