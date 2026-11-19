"""Default averaging for 0/1 labels (INF-39).

f1_score, precision_score and recall_score use average='binary' (the score of
class 1) when y_true and y_pred contain only 0 and 1, and average='micro'
otherwise. An explicit average is respected. For binary F1, a method that is
not used (for example 'wilson') gives a warning.
"""
import warnings

import numpy as np
import pytest
import sklearn.metrics

from infer_ci import MetricEvaluator, accuracy_score, f1_score, precision_score, recall_score

# 0/1 labels where binary scores differ from accuracy:
# TP=4, FP=1, FN=2, TN=3 -> precision 0.8, recall 0.667, F1 0.727, accuracy 0.7
Y_TRUE = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]

Y_TRUE_MULTI = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
Y_PRED_MULTI = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]


@pytest.mark.parametrize("func, sk_func", [
    (f1_score, sklearn.metrics.f1_score),
    (precision_score, sklearn.metrics.precision_score),
    (recall_score, sklearn.metrics.recall_score),
])
def test_binary_labels_default_to_binary_score(func, sk_func):
    expected = sk_func(Y_TRUE, Y_PRED, average='binary')
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default call must not warn
        value, (lower, upper) = func(Y_TRUE, Y_PRED)
    assert value == pytest.approx(expected)
    assert value != pytest.approx(accuracy_score(Y_TRUE, Y_PRED)[0])
    assert lower <= value <= upper


@pytest.mark.parametrize("func", [f1_score, precision_score, recall_score])
def test_numpy_and_bool_binary_labels(func):
    expected = func(Y_TRUE, Y_PRED, average='binary', compute_ci=False)
    assert func(np.array(Y_TRUE), np.array(Y_PRED), compute_ci=False) == pytest.approx(expected)
    assert func(np.array(Y_TRUE, dtype=bool), np.array(Y_PRED, dtype=bool),
                compute_ci=False) == pytest.approx(expected)


def test_evaluate_f1_equals_f1_of_its_precision_and_recall():
    evaluator = MetricEvaluator()
    p, _ = evaluator.evaluate(y_true=Y_TRUE, y_pred=Y_PRED, task='classification',
                              metric='precision', method='wilson')
    r, _ = evaluator.evaluate(y_true=Y_TRUE, y_pred=Y_PRED, task='classification',
                              metric='recall', method='wilson')
    with pytest.warns(UserWarning, match="ignored for binary F1"):
        f1, _ = evaluator.evaluate(y_true=Y_TRUE, y_pred=Y_PRED, task='classification',
                                   metric='f1', method='wilson')
    assert f1 == pytest.approx(2 * p * r / (p + r))
    assert f1 == pytest.approx(sklearn.metrics.f1_score(Y_TRUE, Y_PRED))


@pytest.mark.parametrize("func, sk_func", [
    (f1_score, sklearn.metrics.f1_score),
    (precision_score, sklearn.metrics.precision_score),
    (recall_score, sklearn.metrics.recall_score),
])
def test_multiclass_stays_micro(func, sk_func):
    expected = sk_func(Y_TRUE_MULTI, Y_PRED_MULTI, average='micro')
    value, _ = func(Y_TRUE_MULTI, Y_PRED_MULTI)
    assert value == pytest.approx(expected)
    assert value == pytest.approx(func(Y_TRUE_MULTI, Y_PRED_MULTI, average='micro')[0])


@pytest.mark.parametrize("func", [f1_score, precision_score, recall_score])
@pytest.mark.parametrize("average", ['micro', 'macro'])
def test_explicit_average_is_respected(func, average):
    expected = getattr(sklearn.metrics, func.__name__)(Y_TRUE, Y_PRED, average=average)
    value, _ = func(Y_TRUE, Y_PRED, average=average)
    assert value == pytest.approx(expected)


def test_explicit_micro_on_binary_labels_equals_accuracy():
    value, _ = f1_score(Y_TRUE, Y_PRED, average='micro')
    assert value == pytest.approx(accuracy_score(Y_TRUE, Y_PRED)[0])


@pytest.mark.parametrize("method", ['wilson', 'normal', 'beta'])
def test_warning_when_method_is_ignored_for_binary_f1(method):
    with pytest.warns(UserWarning, match=f"method='{method}' is ignored for binary F1"):
        f1_score(Y_TRUE, Y_PRED, method=method)


@pytest.mark.parametrize("kwargs", [
    {},  # default method 'takahashi' uses the delta method as documented
    {"method": "bootstrap_bca", "n_resamples": 200, "random_state": 0},
    {"average": "micro", "method": "wilson"},
])
def test_no_warning_when_method_is_used(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        f1_score(Y_TRUE, Y_PRED, **kwargs)
