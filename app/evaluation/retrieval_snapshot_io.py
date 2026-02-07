"""
Read/write utilities for deterministic retrieval snapshots.

Motivation:
- Retrieval uses Chroma query_texts, which triggers embedding calls.
- For benchmarking multiple models, we want to pay retrieval/embeddings ONCE,
  persist snapshots to disk, then reuse them across model runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from .benchmark_types import RetrievalSnapshot


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    return sha256_bytes(p.read_bytes())


def json_fingerprint(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def save_retrieval_snapshots(
    path: str | Path,
    snapshots: Dict[str, RetrievalSnapshot],
    meta: Optional[dict] = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    with out.open("w", encoding="utf-8") as f:
        for sample_id, snapshot in snapshots.items():
            row = {
                "sample_id": sample_id,
                "snapshot": snapshot.model_dump(),
                "meta": meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def load_retrieval_snapshots(path: str | Path) -> Tuple[Dict[str, RetrievalSnapshot], Dict[str, object]]:
    """
    Load snapshots from a JSONL file written by save_retrieval_snapshots().

    Returns (snapshots_by_sample_id, meta).
    """
    snapshots: Dict[str, RetrievalSnapshot] = {}
    meta: Dict[str, object] = {}

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            snap_raw = row.get("snapshot")
            if not sample_id or not snap_raw:
                continue
            if not meta and isinstance(row.get("meta"), dict):
                meta = dict(row.get("meta") or {})
            snapshots[sample_id] = RetrievalSnapshot.model_validate(snap_raw)

    return snapshots, meta

