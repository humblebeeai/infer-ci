"""Attendance metrics for the ChokePoint data (INF-50).

Compares what an attendance system recorded with who really passed, and
computes the same metrics as INF-43 with 95% confidence intervals.

Tables
------
records : session, camera, person_id, time_s
    What the system recorded.
truth : session, camera, person_id, start_time_s
    Who really passed (the attendance table from INF-48).

Records are compared only with the truth of the same session and camera.
"""
import numpy as np
import pandas as pd

from infer_ci import accuracy_score, f1_score, mae, mean_bia_dev, precision_score, recall_score
from infer_ci.methods import bootstrap_ci

KEYS = ["session", "camera", "person_id"]
RESULTS = ["match", "late", "early", "missed", "extra"]

# People who were never enrolled (only in Portal 2). They stay in the truth table
# from INF-48 but are not in the denominator of the attendance metrics.
UNENROLLED = ("0002", "0028", "0029", "0030")

# For P/R/F1, a late or early record is a missed real pass (FN) and a wrong record (FP).
EVENTS = {
    "match": [(1, 1)],
    "late": [(1, 0), (0, 1)],
    "early": [(1, 0), (0, 1)],
    "missed": [(1, 0)],
    "extra": [(0, 1)],
}


def records_from_predictions(predictions: pd.DataFrame, threshold: float = 0.5, k: int = 6) -> pd.DataFrame:
    """Build records from face predictions (one row per detected face).

    In each frame only the largest face is kept (the person walking through).
    The face is recognised as person P when best_id = P and best_score >= threshold.
    A person is recorded in a session and camera when they are recognised in at
    least k frames, at the time of the k-th recognised frame.

    predictions needs: session, camera, frame, time_s, box_area, best_id, best_score.
    Read the CSV with dtype={"best_id": str} to keep IDs such as "0014".
    """
    largest = (predictions.sort_values("box_area", ascending=False, kind="stable")
               .drop_duplicates(["session", "camera", "frame"]))
    recognised = largest[largest["best_score"] >= threshold].sort_values(["session", "camera", "frame"])
    recognised = recognised.rename(columns={"best_id": "person_id"})
    kth = recognised.groupby(KEYS, sort=False).nth(k - 1)
    return kth[KEYS + ["time_s"]].sort_values(["session", "camera", "time_s"]).reset_index(drop=True)


def compare(records: pd.DataFrame, truth: pd.DataFrame, tolerance: float = 1.0,
            unenrolled=UNENROLLED) -> pd.DataFrame:
    """Compare records with the truth of the same session and camera.

    Returns one row per result: session, camera, person_id, start_time_s,
    time_s, diff_s (time_s - start_time_s) and result:

    - match: the recorded person passed and |diff_s| <= tolerance
    - late / early: the recorded person passed, but diff_s > tolerance / < -tolerance
    - missed: the person passed but was not recorded
    - extra: the person was recorded but did not pass

    People in ``unenrolled`` are removed from the truth: when they pass and are
    not recorded, nothing is counted. A record with their ID would be extra.

    If the same person has several truth rows or records in one session and
    camera, they are paired in time order and the rest are missed or extra.
    """
    truth = truth[~truth["person_id"].isin(list(unenrolled))]
    t = truth[KEYS + ["start_time_s"]].sort_values("start_time_s").copy()
    r = records[KEYS + ["time_s"]].sort_values("time_s").copy()
    t["n"] = t.groupby(KEYS).cumcount()
    r["n"] = r.groupby(KEYS).cumcount()
    out = t.merge(r, on=KEYS + ["n"], how="outer").drop(columns="n")

    out["diff_s"] = out["time_s"] - out["start_time_s"]
    # A small margin so that a difference of exactly `tolerance` seconds is
    # not lost to floating-point rounding (for example 12.6 - 10.6).
    limit = tolerance + 1e-9
    result = np.select(
        [out["time_s"].isna(), out["start_time_s"].isna(),
         out["diff_s"].abs() <= limit, out["diff_s"] > limit],
        ["missed", "extra", "match", "late"], default="early")
    out["result"] = result
    return out.sort_values(["session", "camera", "start_time_s", "time_s"]).reset_index(drop=True)


def count_results(outcomes: pd.DataFrame) -> dict:
    """Count match / late / early / missed / extra."""
    counts = outcomes["result"].value_counts()
    return {name: int(counts.get(name, 0)) for name in RESULTS}


def p95_abs_diff(y_true, y_pred) -> float:
    """95th percentile of |y_pred - y_true|."""
    return float(np.percentile(np.abs(np.asarray(y_pred) - np.asarray(y_true)), 95))


def attendance_metrics(outcomes: pd.DataFrame, group: str = "all", confidence_level: float = 0.95,
                       n_resamples: int = 2000, random_state: int = 0) -> pd.DataFrame:
    """Metrics with confidence intervals, as in INF-43.

    Returns a table with group, metric, value, lower, upper:

    - Accuracy = match / (match + late + early + missed + extra), Wilson interval
    - Late/early, missed and extra rates, Wilson interval
    - Event precision, recall and F1, average="binary"
    - Timing for match and late/early rows (seconds): MAE, mean bias
      (average time_s - start_time_s) and the 95th percentile of |difference|,
      bootstrap intervals
    """
    res = outcomes["result"]
    ones = np.ones(len(outcomes), dtype=int)
    rates = {
        "Accuracy": res == "match",
        "Late/early rate": res.isin(["late", "early"]),
        "Missed rate": res == "missed",
        "Extra rate": res == "extra",
    }
    values = {name: accuracy_score(ones, flag.astype(int).to_numpy(), confidence_level, method="wilson")
              for name, flag in rates.items()}

    events = np.array([pair for r in res for pair in EVENTS[r]]).reshape(-1, 2)
    y_true, y_pred = events[:, 0], events[:, 1]
    if y_pred.sum() > 0:
        values["Precision"] = precision_score(y_true, y_pred, confidence_level, average="binary", method="wilson")
    if y_true.sum() > 0:
        values["Recall"] = recall_score(y_true, y_pred, confidence_level, average="binary", method="wilson")
    if y_true.sum() > 0 and y_pred.sum() > 0:
        if (y_true == y_pred).all():
            # No FP and no FN: F1 is 1 and the delta-method variance is 0.
            # f1_score fails here because the confusion matrix has one class.
            values["F1"] = (1.0, (1.0, 1.0))
        else:
            values["F1"] = f1_score(y_true, y_pred, confidence_level, average="binary")

    timed = outcomes[res.isin(["match", "late", "early"])]
    if len(timed) >= 2:
        t_true, t_pred = timed["start_time_s"].to_numpy(), timed["time_s"].to_numpy()
        boot = dict(n_resamples=n_resamples, random_state=random_state, show_progress=False)
        values["Timing MAE (s)"] = mae(t_true, t_pred, confidence_level, **boot)
        values["Timing mean bias (s)"] = mean_bia_dev(t_true, t_pred, confidence_level, **boot)
        values["Timing p95 |diff| (s)"] = bootstrap_ci(t_true, t_pred, p95_abs_diff, confidence_level,
                                                        method="bootstrap_percentile", **boot)

    rows = [{"group": group, "metric": name, "value": float(v), "lower": float(lo), "upper": float(hi)}
            for name, (v, (lo, hi)) in values.items()]
    return pd.DataFrame(rows, columns=["group", "metric", "value", "lower", "upper"])
