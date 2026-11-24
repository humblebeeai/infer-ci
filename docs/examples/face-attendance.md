# Face-recognition attendance (ChokePoint)

## The problem

A face-recognition camera at a door records who comes in and when. How often is
the record right, and how sure can we be from a small test? This example measures
it on the [ChokePoint dataset](https://zenodo.org/records/815657) and gives every
metric a 95% confidence interval with `infer-ci`.

## The setup

- **Enrolment:** 25 people from portal 1 (`P1E_S1`, camera C1). Four people who
  appear only in portal 2 (0002, 0028, 0029, 0030) are never enrolled.
- **Test:** portal 2, sequence `P2E_S1`, camera C3, both recording days
  (`P2E_S1.1` with 21 enrolled people, `P2E_S1.2` with 4).
- **Predictions:** InsightFace `buffalo_l` on every frame (2,247 detected faces).
- **Records:** in each frame only the largest face is used. A person is recorded at
  the 6th frame where `best_score ≥ 0.5` for that person.
- **Truth:** the first frame where each person is seen (frame / 30 s).
- **Comparison** (same session and camera, tolerance 1 s): match, late, early,
  missed or extra. The never-enrolled people are not counted: not recording them
  is correct, and a record with their ID would count as extra.

The code and steps are in `notebooks/chokepoint/` (see its `README.md`).

## Results

All 25 enrolled people were recorded on time: 25 match, 0 late, early, missed or extra.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 1.000 | [0.867, 1.000] |
| Late/early rate | 0.000 | [0.000, 0.133] |
| Missed rate | 0.000 | [0.000, 0.133] |
| Extra rate | 0.000 | [0.000, 0.133] |
| Precision | 1.000 | [0.867, 1.000] |
| Recall | 1.000 | [0.867, 1.000] |
| F1 | 1.000 | [1.000, 1.000] |
| Timing MAE (s) | 0.132 | [0.107, 0.191] |
| Timing mean bias (s) | 0.068 | [−0.025, 0.105] |
| Timing p95 of abs. difference (s) | 0.233 | [0.133, 0.500] |

Rates and precision/recall use Wilson intervals, F1 the delta method, and the timing
metrics bootstrap intervals.

**Settings.** With tolerance 1 s, thresholds 0.3–0.5 record everyone for every k
from 1 to 10. From threshold 0.55 some people are missed (accuracy 0.92–0.96).
With threshold 0.5 and k 6, all 25 people match at every tolerance from 0.5 to 5 s.
No setting produced an extra record.

## What the confidence intervals tell us

- Every point estimate is perfect, but the test has only 25 passes. The accuracy
  interval [0.867, 1.000] says the true accuracy could be as low as about 87%.
- For one recording day alone the interval is wider: `P2E_S1.2` has 4 people, and
  its accuracy interval is [0.510, 1.000]. Small tests cannot show that a system is
  reliable, even when they have no errors.
- The F1 interval [1.000, 1.000] has zero width because the delta method gives zero
  variance when there are no errors. Read it together with the precision and recall
  intervals.
- Records come within 0.5 s of the truth, and 95% within about 0.23 s. The timing
  intervals are narrow, so the 1 s tolerance is safe for this camera.
- To narrow the intervals, evaluate more sequences and cameras.
