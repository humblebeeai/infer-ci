"""Tests for attendance_metrics.py with hand-made tables where the answer is known (INF-50).

Run: python -m pytest tests/test_attendance_metrics.py
"""
import pandas as pd
import pytest

from notebooks.chokepoint.attendance_metrics import (
    attendance_metrics, compare, count_results, records_from_predictions)

# 5 people in session S1, camera C1, tolerance 1 s:
# A match (+0.3 s), B late (+4 s), C missed, D early (-3 s), E extra (never passed)
TRUTH = pd.DataFrame({
    "session": ["S1"] * 4,
    "camera": ["C1"] * 4,
    "person_id": ["A", "B", "C", "D"],
    "start_time_s": [10.0, 20.0, 30.0, 40.0],
})
RECORDS = pd.DataFrame({
    "session": ["S1"] * 4,
    "camera": ["C1"] * 4,
    "person_id": ["A", "B", "D", "E"],
    "time_s": [10.3, 24.0, 37.0, 50.0],
})


def results_by_person(outcomes):
    return dict(zip(outcomes["person_id"], outcomes["result"]))


def test_compare_hand_made_table():
    outcomes = compare(RECORDS, TRUTH, tolerance=1.0)
    assert results_by_person(outcomes) == {"A": "match", "B": "late", "C": "missed", "D": "early", "E": "extra"}
    assert count_results(outcomes) == {"match": 1, "late": 1, "early": 1, "missed": 1, "extra": 1}


def test_tolerance_changes_the_result():
    outcomes = compare(RECORDS, TRUTH, tolerance=5.0)
    assert count_results(outcomes) == {"match": 3, "late": 0, "early": 0, "missed": 1, "extra": 1}


def test_only_same_session_and_camera_are_compared():
    # A is recorded on another camera and in another session: missed on C1, extra elsewhere.
    records = pd.DataFrame({"session": ["S1", "S2"], "camera": ["C2", "C1"],
                            "person_id": ["A", "A"], "time_s": [10.0, 10.0]})
    truth = TRUTH[TRUTH["person_id"] == "A"]
    outcomes = compare(records, truth, tolerance=1.0)
    assert count_results(outcomes) == {"match": 0, "late": 0, "early": 0, "missed": 1, "extra": 2}


def test_unenrolled_people_are_not_counted():
    # 0028 passed but was never enrolled: not recorded -> not counted.
    # A record with an unenrolled ID would be extra.
    truth = pd.concat([TRUTH, pd.DataFrame({"session": ["S1"], "camera": ["C1"],
                                            "person_id": ["0028"], "start_time_s": [60.0]})])
    outcomes = compare(RECORDS, truth, tolerance=1.0)
    assert count_results(outcomes) == {"match": 1, "late": 1, "early": 1, "missed": 1, "extra": 1}
    records = pd.concat([RECORDS, pd.DataFrame({"session": ["S1"], "camera": ["C1"],
                                                "person_id": ["0028"], "time_s": [60.0]})])
    assert count_results(compare(records, truth, tolerance=1.0))["extra"] == 2


def test_metrics_hand_made_table():
    outcomes = compare(RECORDS, TRUTH, tolerance=1.0)
    table = attendance_metrics(outcomes, n_resamples=500).set_index("metric")
    # accuracy and rates over 5 rows
    assert table.loc["Accuracy", "value"] == pytest.approx(1 / 5)
    assert table.loc["Late/early rate", "value"] == pytest.approx(2 / 5)
    assert table.loc["Missed rate", "value"] == pytest.approx(1 / 5)
    assert table.loc["Extra rate", "value"] == pytest.approx(1 / 5)
    # events: TP = 1 (A), FP = 3 (B, D, E), FN = 3 (B, C, D)
    assert table.loc["Precision", "value"] == pytest.approx(1 / 4)
    assert table.loc["Recall", "value"] == pytest.approx(1 / 4)
    assert table.loc["F1", "value"] == pytest.approx(1 / 4)
    # timing for A, B, D: differences +0.3, +4, -3
    assert table.loc["Timing MAE (s)", "value"] == pytest.approx((0.3 + 4 + 3) / 3)
    assert table.loc["Timing mean bias (s)", "value"] == pytest.approx((0.3 + 4 - 3) / 3)
    assert table.loc["Timing p95 |diff| (s)", "value"] == pytest.approx(3.9)
    assert (table["lower"] <= table["value"] + 1e-9).all()
    assert (table["value"] <= table["upper"] + 1e-9).all()


def test_records_from_predictions():
    # Frame 2 has two faces: only the larger one (A) is used.
    # With threshold 0.5 and k = 2, A is recognised in frames 1, 2 and 4 -> recorded at frame 2.
    # B is recognised only once (frame 3) -> not recorded.
    pred = pd.DataFrame({
        "session": ["S1"] * 6,
        "camera": ["C1"] * 6,
        "frame": [1, 2, 2, 3, 4, 5],
        "time_s": [0.1, 0.2, 0.2, 0.3, 0.4, 0.5],
        "box_area": [100, 200, 50, 100, 100, 100],
        "best_id": ["A", "A", "B", "B", "A", "B"],
        "best_score": [0.9, 0.8, 0.9, 0.7, 0.6, 0.4],
    })
    records = records_from_predictions(pred, threshold=0.5, k=2)
    assert records.to_dict("records") == [{"session": "S1", "camera": "C1", "person_id": "A", "time_s": 0.2}]


def test_metrics_when_every_record_matches():
    records = TRUTH.rename(columns={"start_time_s": "time_s"})
    records["time_s"] += [0.1, 0.2, -0.1, 0.3]
    outcomes = compare(records, TRUTH, tolerance=1.0)
    table = attendance_metrics(outcomes, n_resamples=500).set_index("metric")
    assert table.loc["Accuracy", "value"] == pytest.approx(1.0)
    assert table.loc["F1", ["value", "lower", "upper"]].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert table.loc["Timing MAE (s)", "value"] == pytest.approx(0.175)
  
