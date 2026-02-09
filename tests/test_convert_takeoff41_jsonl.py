import json
import subprocess
import sys
from pathlib import Path


def test_convert_takeoff41_jsonl_smoke(tmp_path: Path):
    inp = tmp_path / "takeoff41.jsonl"
    inp.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "question": "What is GIR?",
                        "answer": "Glucose infusion rate.",
                        "question_type": "open_ended",
                        "book_title": "BookA",
                        "source_file": "DocA.md",
                        "chunk_index": 10,
                        "relevance_score": 0.9,
                    }
                ),
                json.dumps(
                    {
                        "id": 20,
                        "question": "Dose range?",
                        "answer": "1-2 mg/kg/min.",
                        "question_type": "calculation",
                        "book_title": "BookB",
                        "source_file": "DocB.md",
                        "chunk_index": 99,
                        "relevance_score": 0.1,
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts/convert_takeoff41_jsonl.py"),
        "--in",
        str(inp),
        "--out-dir",
        str(out_dir),
        "--split",
        "holdout",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    out_jsonl = out_dir / "takeoff41_200_holdout.jsonl"
    out_manifest = out_dir / "takeoff41_200_conversion_manifest.json"
    assert out_jsonl.exists()
    assert out_manifest.exists()

    lines = [line for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]

    # Validate strict schema (at least the fields we rely on downstream).
    from app.evaluation.benchmark_types import DatasetSchema

    parsed = [DatasetSchema.model_validate(r) for r in rows]
    assert parsed[0].sample_id == "takeoff41_001"
    assert parsed[1].sample_id == "takeoff41_020"
    assert parsed[0].track.value == "open_ended"
    assert parsed[0].split.value == "holdout"
    assert parsed[0].reference_answer == "Glucose infusion rate."
    assert parsed[0].source_doc is None
    assert parsed[0].metadata.get("raw_id") == 1

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["input"]["row_count"] == 2
    assert manifest["output"]["track"] == "open_ended"

