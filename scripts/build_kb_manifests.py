#!/usr/bin/env python3
"""
Build explicit KB manifest JSON files for leakage-safe benchmarking.

Why:
- "KB-clean" and "KB-max" must be *explicit*, versionable regimes.
- A manifest gives a stable fingerprint and makes runs reproducible.

Outputs (by default):
- data/kb_manifests/kb_max.json
- data/kb_manifests/kb_clean.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_md_files(docs_dir: Path) -> List[Path]:
    return sorted([p for p in docs_dir.glob("*.md") if p.is_file()])


def build_manifest(*, name: str, docs_dir: Path, include: List[Path], exclude: List[str], description: str) -> Dict:
    rel_paths = [str(p.relative_to(docs_dir)) for p in include]
    sha_map = {str(p.relative_to(docs_dir)): _sha256_file(p) for p in include}
    return {
        "name": name,
        "description": description,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(docs_dir.relative_to(PROJECT_ROOT)),
        "included_files": rel_paths,
        "excluded_files": exclude,
        "file_sha256": sha_map,
        "manifest_sha256": hashlib.sha256(json.dumps(rel_paths, sort_keys=True).encode("utf-8")).hexdigest(),
        "generated_by": "scripts/build_kb_manifests.py",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KB regime manifests (kb_clean, kb_max)")
    p.add_argument("--docs-dir", type=str, default="data/documents", help="Directory of KB markdown documents")
    p.add_argument("--out-dir", type=str, default="data/kb_manifests", help="Output directory for manifests")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    docs_dir = (PROJECT_ROOT / args.docs_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_md = _discover_md_files(docs_dir)
    if not all_md:
        raise SystemExit(f"No .md files found in {docs_dir}")

    # KB-max: everything.
    kb_max = build_manifest(
        name="kb_max",
        docs_dir=docs_dir,
        include=all_md,
        exclude=[],
        description="Maximum-coverage KB (appendix only). Includes question banks / study guides.",
    )
    (out_dir / "kb_max.json").write_text(json.dumps(kb_max, indent=2), encoding="utf-8")

    # KB-clean: exclude obvious question-bank/study-guide documents (paper main).
    excluded = [
        "Questions_from_NeoReviews_A_Study_Guide_for_Neonatal_Perinatal_Medicine.md",
    ]
    include_clean = [p for p in all_md if p.name not in set(excluded)]
    kb_clean = build_manifest(
        name="kb_clean",
        docs_dir=docs_dir,
        include=include_clean,
        exclude=excluded,
        description="Leakage-safe KB (paper main). Excludes question-bank / study-guide documents.",
    )
    (out_dir / "kb_clean.json").write_text(json.dumps(kb_clean, indent=2), encoding="utf-8")

    print(f"Built manifests in: {out_dir}")
    print(f"  kb_max:   {len(kb_max['included_files'])} files")
    print(f"  kb_clean: {len(kb_clean['included_files'])} files (excluded: {excluded})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

