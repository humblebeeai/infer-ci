import inspect

import numpy as np
import pytest

import infer_ci
from infer_ci import MetricEvaluator, evaluate_metric

NEW_REGRESSION_METRICS = [
    'med_abs_err',
    'mean_bia_dev',
    'huber_loss',
    'exp_var_score',
    'sym_mean_abs_per_error',
    'rmse_log',
]


def _data():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(1, 10, size=60)
    y_pred = y_true + rng.normal(0, 1, size=60)
    return y_true, y_pred


@pytest.mark.parametrize('metric', NEW_REGRESSION_METRICS)
def test_new_metric_is_listed(metric):
    assert metric in MetricEvaluator().get_available_metrics('regression')


@pytest.mark.parametrize('metric', NEW_REGRESSION_METRICS)
def test_unified_interface_matches_direct_call(metric):
    y_true, y_pred = _data()
    direct = getattr(infer_ci, metric)
    kwargs = dict(confidence_level=0.95, method='bootstrap_percentile',
                  n_resamples=500, random_state=np.random.RandomState(1))
    unified = evaluate_metric(y_true, y_pred, task='regression', metric=metric, **kwargs)
    kwargs['random_state'] = np.random.RandomState(1)
    expected = direct(y_true, y_pred, **kwargs)

    assert unified[0] == pytest.approx(expected[0])
    assert unified[1] == pytest.approx(expected[1])


@pytest.mark.parametrize('metric', NEW_REGRESSION_METRICS)
def test_unified_interface_without_ci(metric):
    y_true, y_pred = _data()
    value = evaluate_metric(y_true, y_pred, task='regression', metric=metric, compute_ci=False)
    assert value == pytest.approx(getattr(infer_ci, metric)(y_true, y_pred, compute_ci=False))


def test_huber_delta_is_passed_through():
    y_true, y_pred = _data()
    value = evaluate_metric(y_true, y_pred, task='regression', metric='huber_loss',
                            compute_ci=False, delta=0.5)
    assert value == pytest.approx(infer_ci.huber_loss(y_true, y_pred, delta=0.5, compute_ci=False))


def test_every_exported_regression_metric_is_registered():
    from infer_ci import regression_metrics as module
    exported = {name for name in infer_ci.__all__
                if callable(getattr(module, name, None))}
    registered = set(MetricEvaluator().regression_metrics.values())
    missing = sorted(name for name in exported
                     if getattr(module, name) not in registered)
    assert missing == []


def test_rmse_log_takes_y_true_first():
    # Same argument order as the other metrics. RMSLE gives the same value
    # either way, so check the signature itself.
    params = list(inspect.signature(infer_ci.rmse_log).parameters)
    assert params[:2] == ['y_true', 'y_pred']


def test_adjusted_r2_is_listed():
    assert 'adjusted_r2_score' in MetricEvaluator().get_available_metrics('regression')


def test_adjusted_r2_unified_interface_matches_direct_call():
    y_true, y_pred = _data()
    kwargs = dict(confidence_level=0.95, method='bootstrap_percentile', n_resamples=500)
    unified = evaluate_metric(y_true, y_pred, task='regression', metric='adjusted_r2_score',
                              num_features=3, random_state=np.random.RandomState(1), **kwargs)
    direct = infer_ci.adjusted_r2_score(3, y_true, y_pred,
                                        random_state=np.random.RandomState(1), **kwargs)
    assert unified[0] == pytest.approx(direct[0])
    assert unified[1] == pytest.approx(direct[1])


def test_adjusted_r2_needs_num_features():
    y_true, y_pred = _data()
    with pytest.raises(ValueError, match='num_features'):
        evaluate_metric(y_true, y_pred, task='regression', metric='adjusted_r2_score',
                        compute_ci=False)


@pytest.mark.parametrize('num_features', [-1, 2.5, True])
def test_adjusted_r2_rejects_invalid_num_features(num_features):
    y_true, y_pred = _data()
    with pytest.raises(ValueError, match='num_features'):
        infer_ci.adjusted_r2_score(num_features, y_true, y_pred, compute_ci=False)


@pytest.mark.parametrize('n_samples, num_features', [(5, 4), (4, 3)])
def test_adjusted_r2_rejects_too_few_samples(n_samples, num_features):
    # The formula divides by n - num_features - 1, so it needs n > num_features + 1.
    y_true, y_pred = _data()
    with pytest.raises(ValueError, match='samples'):
        infer_ci.adjusted_r2_score(num_features, y_true[:n_samples], y_pred[:n_samples],
                                   compute_ci=False)


def test_adjusted_r2_smallest_valid_sample():
    y_true, y_pred = _data()
    value = infer_ci.adjusted_r2_score(3, y_true[:5], y_pred[:5], compute_ci=False)
    assert np.isfinite(value)
