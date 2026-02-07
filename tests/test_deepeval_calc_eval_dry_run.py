import json
import subprocess
import sys
from pathlib import Path


def test_deepeval_calc_eval_dry_run(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "calc_001",
                "track": "open_ended",
                "split": "holdout",
                "question": "Calculate 3 g/kg/day for 2 kg.",
                "reference_answer": "6 g/day.",
                "domain": "tpn",
                "source_doc": "DocA",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = tmp_path / "run_records.jsonl"
    records.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "sample_id": "calc_001",
                "track": "open_ended",
                "model_id": "m1",
                "prompt_strategy": "ZS",
                "rag_enabled": False,
                "response_text": "6 g/day.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts/deepeval_calc_eval.py"),
        "--dataset",
        str(dataset),
        "--records",
        str(records),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "Dry run OK" in proc.stdout

