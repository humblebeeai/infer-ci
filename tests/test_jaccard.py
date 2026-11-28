import pytest
import numpy as np
from sklearn.metrics import jaccard_score as sklearn_jaccard
from infer_ci.binary_metrics import jaccard_score
from infer_ci.evaluator import MetricEvaluator


def test_jaccard_equals_sklearn():
    """1. Random binary cases match sklearn.metrics.jaccard_score"""
    rng = np.random.default_rng(42)
    for _ in range(5):
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)

        our_val = jaccard_score(y_true, y_pred, compute_ci=False)
        sk_val = sklearn_jaccard(y_true, y_pred)
        assert np.isclose(our_val, sk_val)


def test_jaccard_proportion_methods_bounds():
    """2. Proportion methods: lower <= val <= upper and all in [0, 1]"""
    y_true = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    methods = ['wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys', 'binom_test']

    for method in methods:
        val, (lower, upper) = jaccard_score(y_true, y_pred, method=method, compute_ci=True)
        assert 0.0 <= lower <= val <= upper <= 1.0


def test_jaccard_bootstrap_reproducibility():
    """3. Bootstrap reproducible with same random_state"""
    y_true = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]

    val1, ci1 = jaccard_score(y_true, y_pred, method="bootstrap_bca", random_state=42)
    val2, ci2 = jaccard_score(y_true, y_pred, method="bootstrap_bca", random_state=42)

    assert np.isclose(val1, val2)
    assert np.isclose(ci1[0], ci2[0])
    assert np.isclose(ci1[1], ci2[1])


def test_jaccard_compute_ci_false():
    """4. compute_ci=False returns float value only"""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]

    val_prop = jaccard_score(y_true, y_pred, method="wilson", compute_ci=False)
    val_boot = jaccard_score(y_true, y_pred, method="bootstrap_bca", compute_ci=False, random_state=42)

    assert isinstance(val_prop, float)
    assert isinstance(val_boot, float)
    assert np.isclose(val_prop, val_boot)


def test_jaccard_evaluator_direct_call_equality():
    """5. evaluate_metric direct call equality"""
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 1]

    evaluator = MetricEvaluator()
    eval_val, eval_ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task="classification",
        metric="jaccard",
        method="wilson"
    )
    direct_val, direct_ci = jaccard_score(y_true, y_pred, method="wilson")

    assert np.isclose(eval_val, direct_val)
    assert np.isclose(eval_ci[0], direct_ci[0])
    assert np.isclose(eval_ci[1], direct_ci[1])


def test_jaccard_invalid_inputs():
    """6. Clear errors for empty, non-binary, or zero denominator inputs"""
    # Empty
    with pytest.raises(ValueError, match="must not be empty"):
        jaccard_score([], [])

    # Non-binary
    with pytest.raises(ValueError, match="binary values"):
        jaccard_score([0, 1, 2], [0, 1, 1])

    # TP + FP + FN = 0
    with pytest.raises(ValueError, match="TP \+ FP \+ FN is 0"):
        jaccard_score([0, 0, 0], [0, 0, 0])
