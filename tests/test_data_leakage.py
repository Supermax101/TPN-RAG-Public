from app.evaluation.benchmark_types import DatasetSchema, DatasetSplit, DatasetTrack
from app.evaluation.data_leakage import check_data_leakage


def test_data_leakage_detection():
    records = [
        DatasetSchema(
            sample_id="s1",
            track=DatasetTrack.OPEN_ENDED,
            split=DatasetSplit.TRAIN,
            question="What is TPN?",
            reference_answer="A",
        ),
        DatasetSchema(
            sample_id="s2",
            track=DatasetTrack.OPEN_ENDED,
            split=DatasetSplit.HOLDOUT,
            question="What is TPN?",
            reference_answer="B",
        ),
    ]
    report = check_data_leakage(records)
    assert report["leakage_detected"] is True
    assert report["question_overlap_count"] == 1
