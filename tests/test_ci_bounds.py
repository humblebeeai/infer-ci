import numpy as np
import pytest

from infer_ci import f1_score, precision_score, recall_score, roc_auc_score


def event_vectors(tp, fp, fn, tn=0):
    """Labels from counts: TP -> (1, 1), FP -> (0, 1), FN -> (1, 0), TN -> (0, 0)."""
    y_true = np.r_[np.ones(tp), np.zeros(fp), np.ones(fn), np.zeros(tn)].astype(int)
    y_pred = np.r_[np.ones(tp), np.ones(fp), np.zeros(fn), np.zeros(tn)].astype(int)
    return y_true, y_pred


@pytest.mark.parametrize('metric', [f1_score, precision_score, recall_score])
@pytest.mark.parametrize('average', ['binary', 'macro', 'micro'])
# Both classes appear in y_true, so macro averages are defined.
@pytest.mark.parametrize('counts', [(14, 3, 1, 0), (19, 0, 1, 2), (1, 9, 9, 5)])
def test_takahashi_ci_stays_in_unit_interval(metric, average, counts):
    y_true, y_pred = event_vectors(*counts)
    if average == 'binary' and metric is not f1_score:
        pytest.skip('binary precision/recall do not support method="takahashi" (they use proportion CIs)')
    value, (lower, upper) = metric(y_true, y_pred, average=average, method='takahashi')
    assert 0.0 <= lower <= value <= upper <= 1.0


def test_small_event_window_regression():
    # 14 TP, 3 FP, 1 FN: before the covariance fix (diag(p) - p p^T) the
    # interval was [0.710, 1.040]. The correct delta-method interval is
    # [0.753, 0.997], so no clipping is needed here.
    y_true, y_pred = event_vectors(14, 3, 1)
    value, (lower, upper) = f1_score(y_true, y_pred, average='binary', method='takahashi')
    assert value == pytest.approx(0.875)
    assert lower == pytest.approx(0.7535, abs=1e-3)
    assert upper == pytest.approx(0.9965, abs=1e-3)


def test_clip_near_one():
    # 30 TP, 1 FN: the correct delta-method interval is [0.952, 1.016],
    # so the upper bound must be clipped to 1.
    y_true, y_pred = event_vectors(30, 0, 1)
    value, (lower, upper) = f1_score(y_true, y_pred, average='binary', method='takahashi')
    assert value == pytest.approx(0.9836, abs=1e-4)
    assert lower == pytest.approx(0.9515, abs=1e-3)
    assert upper == 1.0


def test_delong_auc_ci_stays_in_unit_interval():
    y_true = np.array([0] * 10 + [1] * 10)
    scores = np.r_[np.linspace(0.0, 0.4, 10), np.linspace(0.45, 1.0, 10)]
    value, (lower, upper) = roc_auc_score(y_true, scores, method='delong')
    assert 0.0 <= lower <= value <= upper <= 1.0

