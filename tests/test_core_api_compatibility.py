"""Offline core API contracts for the INF-29 Python 3.13.2 workstream.

No external datasets, model downloads, plotting windows, or empirical coverage claims.
"""
import math

import numpy as np
import pytest
from sklearn import metrics as reference

from infer_ci import MetricEvaluator, accuracy_score, f1_score, mae, mse, r2_score, rmse


@pytest.fixture
def binary_data():
    return ([0, 1, 1, 0, 1, 0, 1, 0] * 10,
            [0, 1, 0, 0, 1, 1, 1, 0] * 10)


@pytest.fixture
def regression_data():
    return (np.array([3., -.5, 2., 7., 5., 4., 9., -2.] * 10),
            np.array([2.5, 0., 2., 8., 4.5, 3., 8., -1.5] * 10))


@pytest.mark.parametrize('array_input', [False, True])
def test_wilson_matches_independent_formula(binary_data, array_input):
    truth, pred = binary_data
    if array_input:
        truth, pred = np.asarray(truth), np.asarray(pred)
    score, (lower, upper) = MetricEvaluator().evaluate(
        truth, pred, task='classification', metric='accuracy', method='wilson')
    n = len(truth)
    p = sum(a == b for a, b in zip(truth, pred)) / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / denominator
    assert score == pytest.approx(p)
    assert (lower, upper) == pytest.approx((center-radius, center+radius))
    assert 0 <= lower <= score <= upper <= 1


def test_wilson_confidence_level_changes_interval(binary_data):
    truth, pred = binary_data
    value90, ci90 = accuracy_score(truth, pred, confidence_level=.90)
    value99, ci99 = accuracy_score(truth, pred, confidence_level=.99)
    assert value90 == value99
    assert ci99[0] < ci90[0] < ci90[1] < ci99[1]


@pytest.mark.parametrize('metric,expected', [
    ('accuracy', lambda a,b: reference.accuracy_score(a,b)),
    ('precision', lambda a,b: reference.precision_score(a,b)),
    ('recall', lambda a,b: reference.recall_score(a,b)),
    ('f1', lambda a,b: reference.f1_score(a,b,average='micro')),
])
def test_classification_point_scores(binary_data, metric, expected):
    truth, pred = binary_data
    value = MetricEvaluator().evaluate(truth, pred, task='classification',
        metric=metric, method='wilson', compute_ci=False)
    assert np.isscalar(value)
    assert value == pytest.approx(expected(truth,pred), abs=1e-7)


@pytest.mark.parametrize('average', ['binary', 'macro', 'micro'])
def test_direct_f1_averages(binary_data, average):
    truth,pred=binary_data
    value=f1_score(truth,pred,average=average,compute_ci=False)
    assert value == pytest.approx(reference.f1_score(truth,pred,average=average))


@pytest.mark.parametrize('name,direct,expected', [
    ('mae',mae,reference.mean_absolute_error),
    ('mse',mse,reference.mean_squared_error),
    ('rmse',rmse,lambda a,b: math.sqrt(reference.mean_squared_error(a,b))),
    ('r2',r2_score,reference.r2_score),
])
def test_regression_point_scores(regression_data,name,direct,expected):
    truth,pred=regression_data
    value=MetricEvaluator().evaluate(truth,pred,task='regression',metric=name,
        method='jackknife',compute_ci=False)
    assert np.isscalar(value)
    assert value == pytest.approx(expected(truth,pred))
    assert direct(truth,pred,compute_ci=False) == pytest.approx(expected(truth,pred))


@pytest.mark.parametrize('method',['bootstrap_bca','bootstrap_percentile','bootstrap_basic'])
def test_regression_bootstrap_reproducible(regression_data,method):
    truth,pred=regression_data
    evaluator=MetricEvaluator()
    kwargs=dict(task='regression',metric='mae',method=method,n_resamples=400,random_state=42)
    first=evaluator.evaluate(truth,pred,**kwargs)
    second=evaluator.evaluate(truth,pred,**kwargs)
    assert first == second
    score,ci=first
    assert score == pytest.approx(reference.mean_absolute_error(truth,pred))
    assert all(math.isfinite(v) for v in ci)
    assert ci[0] <= score <= ci[1]


@pytest.mark.parametrize('change,error',[
    ({'task':'unknown'},'Unknown task'),
    ({'metric':'unknown'},'Unknown classification metric'),
    ({'method':'unknown'},'Unknown classification method'),
])
def test_invalid_options_report_errors(binary_data,change,error):
    truth,pred=binary_data
    kwargs=dict(task='classification',metric='accuracy',method='wilson')
    kwargs.update(change)
    with pytest.raises(ValueError,match=error):
        MetricEvaluator().evaluate(truth,pred,**kwargs)


def test_missing_prediction_reports_error():
    with pytest.raises(ValueError,match='require'):
        MetricEvaluator().evaluate(y_true=[0,1],task='classification',metric='accuracy')
