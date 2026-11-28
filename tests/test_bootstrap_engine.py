import time

import numpy as np
import pytest
from sklearn import metrics as skm

from infer_ci import bootstrap_ci
from infer_ci.bootstrap_engine import (
    bootstrap_binary_metric,
    bootstrap_binary_metrics,
    bootstrap_weighted,
    iter_resample_indices,
    iter_resample_weights,
    percentile_interval,
    resolve_seed,
)

SKLEARN = {
    "accuracy": skm.accuracy_score,
    "precision": lambda t, p: skm.precision_score(t, p, zero_division=np.nan),
    "recall": lambda t, p: skm.recall_score(t, p, zero_division=np.nan),
    "f1": lambda t, p: skm.f1_score(t, p, zero_division=np.nan),
}


def make_data(n=400, prevalence=0.2, agreement=0.85, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.where(rng.random(n) < agreement, y, 1 - y)
    return y, p


def test_draws_do_not_depend_on_chunk_size():
    whole = np.concatenate(list(iter_resample_indices(37, 1000, seed=7, chunk_size=1000)))
    split = np.concatenate(list(iter_resample_indices(37, 1000, seed=7, chunk_size=33)))
    assert whole.shape == (1000, 37)
    assert np.array_equal(whole, split)

    y, p = make_data()
    a = bootstrap_binary_metric(y, p, "f1", n_resamples=1500, seed=5, chunk_size=1500)
    b = bootstrap_binary_metric(y, p, "f1", n_resamples=1500, seed=5, chunk_size=64)
    assert np.array_equal(a.distribution, b.distribution)
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)


def test_same_seed_same_result_and_seed_is_recorded():
    y, p = make_data()
    a = bootstrap_binary_metric(y, p, "f1", n_resamples=2000, seed=42)
    b = bootstrap_binary_metric(y, p, "f1", n_resamples=2000, seed=42)
    c = bootstrap_binary_metric(y, p, "f1", n_resamples=2000, seed=43)
    assert a.seed == 42
    assert np.array_equal(a.distribution, b.distribution)
    assert not np.array_equal(a.distribution, c.distribution)


def test_seed_none_records_a_seed_that_reproduces_the_run():
    y, p = make_data()
    first = bootstrap_binary_metric(y, p, "recall", n_resamples=500)
    again = bootstrap_binary_metric(y, p, "recall", n_resamples=500, seed=first.seed)
    assert isinstance(first.seed, int)
    assert np.array_equal(first.distribution, again.distribution)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_point_estimates_match_sklearn(seed):
    y, p = make_data(seed=seed)
    results = bootstrap_binary_metrics(y, p, n_resamples=10, seed=0)
    for name, fn in SKLEARN.items():
        assert results[name].estimate == pytest.approx(fn(y, p), abs=1e-12)


def test_resample_values_match_a_naive_loop_over_the_same_draws():
    y, p = make_data(n=60)
    n_resamples = 300
    res = bootstrap_binary_metrics(y, p, n_resamples=n_resamples, seed=11, chunk_size=70)
    draws = np.concatenate(list(iter_resample_indices(len(y), n_resamples, seed=11)))
    for name, fn in SKLEARN.items():
        naive = np.array([fn(y[idx], p[idx]) for idx in draws])
        np.testing.assert_allclose(res[name].distribution, naive, rtol=0, atol=1e-12,
                                   equal_nan=True)


def test_interval_agrees_with_scipy_percentile_bootstrap():
    y, p = make_data(n=300, seed=4)
    ours = bootstrap_binary_metric(y, p, "f1", n_resamples=4000, seed=1)
    _, (lo, hi) = bootstrap_ci(y, p, SKLEARN["f1"], n_resamples=4000,
                               method="bootstrap_percentile", random_state=1,
                               show_progress=False)
    # Different random streams, so only agreement within Monte Carlo error.
    assert ours.ci_lower == pytest.approx(lo, abs=0.02)
    assert ours.ci_upper == pytest.approx(hi, abs=0.02)
    assert ours.ci_lower < ours.estimate < ours.ci_upper


def test_percentile_interval_uses_linear_quantiles_and_skips_nan():
    values = np.array([np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, np.inf])
    lo, hi, n_valid = percentile_interval(values, confidence=0.5)
    assert n_valid == 5
    assert (lo, hi) == (1.0, 3.0)
    assert percentile_interval(np.array([np.nan, np.nan]), 0.95)[2] == 0


def test_undefined_metric_is_nan_and_never_zero():
    # No positive predictions: precision and F1 have a zero denominator.
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.zeros(6, dtype=int)
    res = bootstrap_binary_metrics(y, p, metrics=("precision", "recall", "f1"),
                                   n_resamples=200, seed=0)
    assert np.isnan(res["precision"].estimate)
    assert np.isnan(res["precision"].ci_lower) and np.isnan(res["precision"].ci_upper)
    assert res["precision"].n_valid == 0
    assert res["recall"].estimate == 0.0
    # F1 = 2TP / (2TP + FP + FN) is defined (= 0) whenever there is a positive label.
    assert res["f1"].estimate == 0.0


def test_partially_valid_resamples_are_counted():
    # A single positive prediction: many resamples draw no positive prediction.
    y, p = make_data(n=40, seed=3)
    p = np.zeros_like(p)
    p[np.flatnonzero(y)[0]] = 1
    res = bootstrap_binary_metric(y, p, "precision", n_resamples=2000, seed=0)
    assert res.estimate == 1.0
    assert 0 < res.n_valid < 2000
    assert np.isfinite(res.distribution).sum() == res.n_valid


def test_pos_label_and_string_labels():
    y, p = make_data(seed=8)
    flipped = bootstrap_binary_metric(1 - y, 1 - p, "recall", n_resamples=50, seed=0,
                                      pos_label=0)
    assert flipped.estimate == pytest.approx(skm.recall_score(y, p))

    ys = np.where(y == 1, "defect", "ok")
    ps = np.where(p == 1, "defect", "ok")
    as_str = bootstrap_binary_metric(ys, ps, "f1", n_resamples=300, seed=2, pos_label="defect")
    as_int = bootstrap_binary_metric(y, p, "f1", n_resamples=300, seed=2)
    assert np.array_equal(as_str.distribution, as_int.distribution, equal_nan=True)


def test_groups_weight_each_row_by_its_group_count():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    for weights, idx in zip(iter_resample_weights(6, 100, seed=9, groups=groups),
                            iter_resample_indices(3, 100, seed=9)):
        counts = np.stack([np.bincount(row, minlength=3) for row in idx])  # a, b, c
        expected = counts[:, [0, 0, 1, 2, 2, 2]]
        assert np.array_equal(weights, expected)


def test_one_row_per_group_equals_row_level_resampling():
    y, p = make_data(n=50)
    rows = bootstrap_binary_metric(y, p, "accuracy", n_resamples=400, seed=6)
    grouped = bootstrap_binary_metric(y, p, "accuracy", n_resamples=400, seed=6,
                                      groups=np.arange(50))
    assert np.array_equal(rows.distribution, grouped.distribution)


def test_bootstrap_weighted_supports_multi_output_statistics():
    x = np.arange(10, dtype=float)

    def stat(w):
        total = w.sum(axis=1)
        return np.column_stack([w @ x / total, total])

    est, dist, seed = bootstrap_weighted(stat, 10, n_resamples=25, seed=3)
    assert est.tolist() == [4.5, 10.0]
    assert dist.shape == (25, 2)
    assert np.all(dist[:, 1] == 10)  # each resample draws n rows
    assert seed == 3


@pytest.mark.parametrize("kwargs, error", [
    (dict(y_true=[0, 1, 2], y_pred=[0, 1, 1]), ValueError),
    (dict(y_true=[0, 1], y_pred=[0, 1, 1]), ValueError),
    (dict(y_true=[], y_pred=[]), ValueError),
    (dict(y_true=[0, 1], y_pred=[0, 1], metric="auc"), ValueError),
    (dict(y_true=[0, 1], y_pred=[0, 1], confidence=1.0), ValueError),
    (dict(y_true=[0, 1], y_pred=[0, 1], n_resamples=0), ValueError),
    (dict(y_true=[0, 1], y_pred=[0, 1], seed=1.5), TypeError),
    (dict(y_true=[0, 1], y_pred=[0, 1], seed=-1), ValueError),
])
def test_invalid_input(kwargs, error):
    with pytest.raises(error):
        bootstrap_binary_metric(**kwargs)


def test_resolve_seed():
    assert resolve_seed(7) == 7
    assert resolve_seed(np.int64(7)) == 7
    with pytest.raises(TypeError):
        resolve_seed(True)


def test_fresh_seed_is_safe_for_json_and_javascript():
    # JavaScript parses JSON numbers as doubles; seeds must stay well below 2**53.
    assert all(0 <= resolve_seed(None) < 2**32 for _ in range(50))


def test_pos_label_not_among_labels_raises():
    with pytest.raises(ValueError, match="pos_label"):
        bootstrap_binary_metric(["ok", "defect"], ["ok", "defect"], "recall", n_resamples=5)
    # A single label is allowed: everything is simply negative.
    res = bootstrap_binary_metric(["ok", "ok"], ["ok", "ok"], "specificity",
                                  n_resamples=5, seed=0, pos_label="defect")
    assert res.estimate == 1.0


def test_results_compare_by_identity():
    y, p = make_data()
    a = bootstrap_binary_metric(y, p, "f1", n_resamples=20, seed=0)
    b = bootstrap_binary_metric(y, p, "f1", n_resamples=20, seed=0)
    assert a == a and a != b  # must not raise "truth value of an array is ambiguous"


def test_automatic_chunk_size_gives_the_same_draws():
    y, p = make_data(n=20_000, seed=1)  # automatic chunk: 4_000_000 // 20_000 = 200
    auto = bootstrap_binary_metric(y, p, "f1", n_resamples=450, seed=3)
    fixed = bootstrap_binary_metric(y, p, "f1", n_resamples=450, seed=3, chunk_size=500)
    assert np.array_equal(auto.distribution, fixed.distribution)


def test_grouped_binary_path_matches_row_weight_path():
    y, p = make_data(n=300, seed=2)
    groups = np.random.default_rng(0).integers(0, 40, size=300)
    fast = bootstrap_binary_metrics(y, p, n_resamples=500, seed=4, groups=groups)
    names = list(fast)

    def stat(weights):
        from infer_ci.bootstrap_engine import binary_metrics_from_counts, confusion_indicators
        return binary_metrics_from_counts(weights @ confusion_indicators(y, p), names)

    est, dist, _ = bootstrap_weighted(stat, 300, n_resamples=500, seed=4, groups=groups)
    for j, name in enumerate(names):
        assert fast[name].estimate == pytest.approx(est[j], abs=1e-12)
        np.testing.assert_allclose(fast[name].distribution, dist[:, j], atol=1e-12,
                                   equal_nan=True)


def test_progress_with_bad_chunk_size_raises_value_error():
    with pytest.raises(ValueError, match="chunk_size"):
        bootstrap_weighted(lambda w: w.sum(axis=1), 5, n_resamples=10, seed=0,
                           chunk_size=0, show_progress=True)


def test_peak_memory_is_bounded_for_large_inputs():
    import tracemalloc

    y, p = make_data(n=200_000, prevalence=0.1, seed=0)
    tracemalloc.start()
    try:
        bootstrap_binary_metrics(y, p, n_resamples=200, seed=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 400e6


@pytest.mark.slow
def test_runtime_on_largest_prototype_input():
    y, p = make_data(n=10_000, prevalence=0.1, seed=0)
    start = time.perf_counter()
    bootstrap_binary_metrics(y, p, n_resamples=9999, seed=0)
    assert time.perf_counter() - start < 60
