# ChokePoint Face Attendance Evaluation (`P2E_S1_C3`)

## 1. The Problem
This example evaluates how closely predicted face attendance/identification outputs match ground-truth identity labels on the ChokePoint sequence `P2E_S1_C3`. Instead of re-running heavy video frame inference, it compares parsed ground-truth annotations against precomputed prediction CSVs (`P2E_S1_C3.csv`) to compute the match rate and its 95% confidence interval.

## 2. The Setup
- **Dataset & Sequence:** ChokePoint `P2E_S1_C3`
- **Inputs:** Ground-truth archive `groundtruth.tar.xz` (~`0.5 MB`) and prediction file `P2E_S1_C3.csv`
- **Metric:** GT ID Match Rate
- **Execution:** Run via `run_eval.py` or `chokepoint_attendance.ipynb` in a Python 3.10 virtual environment (`.venv`)

## 3. Results Table

| Sequence | Metric | Samples (N) | Rate | 95% CI Lower | 95% CI Upper |
| :--- | :--- | ---: | ---: | ---: | ---: |
| P2E_S1_C3 | GT ID Match Rate | 2247 | 0.7583 | 0.7406 | 0.776 |

## 4. What the Confidence Intervals Tell Us
- Across `2,247` evaluated samples, the point estimate is `0.7583` (`75.83%`) with a 95% confidence interval of `[0.7406, 0.7760]` (`74.06%` to `77.60%`).
- Because the sample size (`N = 2,247`) is relatively large, the interval width is narrow (about `0.0354`, or `3.54` percentage points), indicating a well-bounded estimate around ~`75.8%`.
- At the same time, the upper bound around `77.6%` shows that roughly a quarter of samples still fail to match ground-truth IDs, highlighting room for model and matching improvements before production attendance use.
