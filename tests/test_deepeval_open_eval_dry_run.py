import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_deepeval_open_eval_dry_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    records_path = tmp_path / "run_records_20260101_000000.jsonl"
    out_dir = tmp_path / "out"

    _write_jsonl(
        dataset_path,
        [
            {
                "sample_id": "s1",
                "track": "open_ended",
                "split": "holdout",
                "question": "What is cyclic TPN and when is it used?",
                "reference_answer": "Cyclic TPN is PN infused over fewer hours (e.g., 12-18h) to allow time off the pump.",
                "domain": "tpn",
                "source_doc": None,
                "metadata": {},
            },
            {
                "sample_id": "s2",
                "track": "open_ended",
                "split": "holdout",
                "question": "What is a typical maximum GIR in pediatrics?",
                "reference_answer": "A typical maximum GIR is around 12-14 mg/kg/min depending on age and tolerance.",
                "domain": "tpn",
                "source_doc": None,
                "metadata": {},
            },
        ],
    )

    _write_jsonl(
        records_path,
        [
            {
                "run_id": "r1",
                "sample_id": "s1",
                "track": "open_ended",
                "model_id": "phi-4",
                "model_name": "microsoft/phi-4",
                "provider": "huggingface",
                "model_tier": "open",
                "prompt_strategy": "ZS",
                "rag_enabled": False,
                "repeat_index": 0,
                "question": "What is cyclic TPN and when is it used?",
                "prompt": "Q: ...",
                "response_text": "Final answer: Cyclic TPN means infusing PN over fewer hours to give time off.",
                "parsed_answer": None,
                "correct": None,
                "retrieval_context_hash": None,
                "retrieval_snapshot_id": None,
                "latency_ms": 1.0,
                "tokens_used": 0,
                "metrics": {},
                "structured_output_used": False,
                "error": None,
            },
            {
                "run_id": "r2",
                "sample_id": "s2",
                "track": "open_ended",
                "model_id": "phi-4",
                "model_name": "microsoft/phi-4",
                "provider": "huggingface",
                "model_tier": "open",
                "prompt_strategy": "ZS",
                "rag_enabled": False,
                "repeat_index": 0,
                "question": "What is a typical maximum GIR in pediatrics?",
                "prompt": "Q: ...",
                "response_text": "Final answer: Often ~12-14 mg/kg/min depending on age and tolerance.",
                "parsed_answer": None,
                "correct": None,
                "retrieval_context_hash": None,
                "retrieval_snapshot_id": None,
                "latency_ms": 1.0,
                "tokens_used": 0,
                "metrics": {},
                "structured_output_used": False,
                "error": None,
            },
        ],
    )

    script = Path(__file__).resolve().parent.parent / "scripts" / "deepeval_open_eval.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rubric",
            "qanda20",
            "--dataset",
            str(dataset_path),
            "--records",
            str(records_path),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dry run OK" in (proc.stdout + proc.stderr)

