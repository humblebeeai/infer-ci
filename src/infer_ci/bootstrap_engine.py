"""Vectorized bootstrap engine.

The engine resamples *units* (rows, or groups such as persons) with replacement
and turns each resample into a vector of per-row weights: a row's weight is the
number of times its unit was drawn. Any statistic that can be computed from
weighted counts (confusion-matrix cells, error counts, pair counts) is then
evaluated for a whole chunk of resamples with one matrix product instead of one
Python call per resample.

Resamples are drawn in chunks to keep memory bounded: a chunk of ``b`` resamples
over ``n`` rows holds a ``(b, n)`` weight matrix.

Reproducibility
---------------
The seed actually used is always returned. With the same seed, input and
``n_resamples``, the draws and therefore the interval are identical, whatever
the ``chunk_size``. Draws come from ``numpy.random.default_rng(seed)`` (PCG64).

Interval
--------
Two-sided percentile interval: the ``alpha/2`` and ``1 - alpha/2`` quantiles of
the valid (finite) bootstrap values, computed with ``numpy.quantile`` using the
``"linear"`` method (type 7 in Hyndman & Fan, numpy's default). A resample where
the statistic is undefined (for example precision when no positive predictions
were drawn) is NaN and is excluded; ``n_valid`` reports how many were kept.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

QUANTILE_METHOD = "linear"
DEFAULT_CHUNK_SIZE = 500

SeedLike = Optional[Union[int, np.integer]]

#: Binary metrics computable from confusion counts (tp, fp, fn, tn).
BINARY_METRICS = ("accuracy", "precision", "recall", "f1",
                  "specificity", "npv", "fpr")


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a bootstrap run for one statistic.

    ``estimate``, ``ci_lower`` and ``ci_upper`` are ``nan`` when undefined: the
    estimate when the statistic is undefined on the original sample, the
    bounds when no resample produced a valid value.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    distribution: np.ndarray
    n_valid: int
    n_resamples: int
    confidence: float
    seed: int
    method: str = "bootstrap_percentile"
    quantile_method: str = QUANTILE_METHOD


def resolve_seed(seed: SeedLike = None) -> int:
    """Return a concrete integer seed, drawing fresh entropy when ``seed`` is None."""
    if seed is None:
        return int(np.random.SeedSequence().entropy)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def _unit_index(n_rows: int, groups: Optional[Sequence]) -> Tuple[np.ndarray, int]:
    """Map each row to a resampling unit; return (row -> unit index, number of units)."""
    if groups is None:
        return np.arange(n_rows), n_rows
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError("groups must be one-dimensional with one entry per row")
    _, row_unit = np.unique(groups, return_inverse=True)
    return row_unit.ravel(), int(row_unit.max()) + 1


def iter_resample_indices(n_units: int,
                          n_resamples: int,
                          seed: int,
                          chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[np.ndarray]:
    """Yield ``(b, n_units)`` arrays of drawn unit indices, chunk by chunk.

    Each row of a chunk is one resample: ``n_units`` units drawn uniformly with
    replacement. Exposed so a reference implementation can replay the exact
    same draws.
    """
    if n_units < 1:
        raise ValueError("need at least one unit to resample")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_resamples:
        b = min(chunk_size, n_resamples - done)
        # One integer per (resample, draw), consumed in row-major order, so the
        # stream does not depend on how resamples are split into chunks.
        yield rng.integers(0, n_units, size=b * n_units).reshape(b, n_units)
        done += b


def unit_counts(indices: np.ndarray, n_units: int) -> np.ndarray:
    """Convert a ``(b, k)`` array of drawn indices into ``(b, n_units)`` draw counts."""
    b = indices.shape[0]
    offsets = (np.arange(b) * n_units)[:, None]
    return np.bincount((indices + offsets).ravel(),
                       minlength=b * n_units).reshape(b, n_units)


def iter_resample_weights(n_rows: int,
                          n_resamples: int,
                          seed: int,
                          groups: Optional[Sequence] = None,
                          chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[np.ndarray]:
    """Yield ``(b, n_rows)`` float weight matrices, one row per resample.

    Without ``groups`` a row's weight is how many times that row was drawn.
    With ``groups`` the units are the distinct group labels: as many groups as
    exist are drawn with replacement, and every row gets its group's count.
    """
    row_unit, n_units = _unit_index(n_rows, groups)
    for indices in iter_resample_indices(n_units, n_resamples, seed, chunk_size):
        counts = unit_counts(indices, n_units)
        weights = counts if groups is None else counts[:, row_unit]
        yield weights.astype(np.float64)


def percentile_interval(values: np.ndarray, confidence: float) -> Tuple[float, float, int]:
    """Percentile interval over the finite entries of ``values``; returns (lo, hi, n_valid)."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    alpha = 1.0 - confidence
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)
    return float(lo), float(hi), int(valid.size)


def bootstrap_weighted(statistic: Callable[[np.ndarray], np.ndarray],
                       n_rows: int,
                       n_resamples: int = 9999,
                       confidence: float = 0.95,
                       seed: SeedLike = None,
                       groups: Optional[Sequence] = None,
                       chunk_size: int = DEFAULT_CHUNK_SIZE,
                       show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """Evaluate ``statistic`` on the original sample and on every resample.

    ``statistic`` receives a ``(b, n_rows)`` weight matrix and returns an array
    of shape ``(b,)`` or ``(b, k)`` (NaN where undefined). The original sample
    is evaluated with all weights equal to 1.

    Returns ``(estimate, distribution, seed)`` where ``estimate`` has shape
    ``(k,)`` (or ``()``), ``distribution`` has shape ``(n_resamples, k)`` (or
    ``(n_resamples,)``) and ``seed`` is the seed actually used.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    used_seed = resolve_seed(seed)
    estimate = np.asarray(statistic(np.ones((1, n_rows))), dtype=np.float64)[0]

    chunks = iter_resample_weights(n_rows, n_resamples, used_seed, groups, chunk_size)
    if show_progress:
        from tqdm import tqdm
        chunks = tqdm(chunks, total=-(-n_resamples // chunk_size), desc="Bootstrap")
    parts = [np.asarray(statistic(w), dtype=np.float64) for w in chunks]
    return estimate, np.concatenate(parts, axis=0), used_seed


def _check_binary(y_true, y_pred, pos_label) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional and of equal length")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size > 2:
        raise ValueError(f"binary metrics need at most two labels, got {labels.tolist()}")
    return y_true == pos_label, y_pred == pos_label


def confusion_indicators(y_true, y_pred, pos_label=1) -> np.ndarray:
    """``(n, 4)`` float indicator matrix with columns tp, fp, fn, tn."""
    t, p = _check_binary(y_true, y_pred, pos_label)
    return np.column_stack([t & p, ~t & p, t & ~p, ~t & ~p]).astype(np.float64)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def binary_metrics_from_counts(counts: np.ndarray, metrics: Sequence[str]) -> np.ndarray:
    """Binary metrics from ``(b, 4)`` weighted counts (tp, fp, fn, tn); NaN where undefined."""
    tp, fp, fn, tn = (counts[:, i] for i in range(4))
    formulas: Dict[str, Callable[[], np.ndarray]] = {
        "accuracy": lambda: _ratio(tp + tn, tp + fp + fn + tn),
        "precision": lambda: _ratio(tp, tp + fp),
        "recall": lambda: _ratio(tp, tp + fn),
        "f1": lambda: _ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": lambda: _ratio(tn, tn + fp),
        "npv": lambda: _ratio(tn, tn + fn),
        "fpr": lambda: _ratio(fp, fp + tn),
    }
    unknown = [m for m in metrics if m not in formulas]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; choose from {list(BINARY_METRICS)}")
    return np.column_stack([formulas[m]() for m in metrics])


def bootstrap_binary_metrics(y_true,
                             y_pred,
                             metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1"),
                             n_resamples: int = 9999,
                             confidence: float = 0.95,
                             seed: SeedLike = None,
                             groups: Optional[Sequence] = None,
                             pos_label=1,
                             chunk_size: int = DEFAULT_CHUNK_SIZE,
                             show_progress: bool = False) -> Dict[str, BootstrapResult]:
    """Percentile bootstrap CIs for binary metrics, all from the same resamples.

    Metrics are computed directly from confusion counts with ``pos_label`` as
    the positive class: no averaging, and no small constant added to any
    denominator. A metric that is undefined (zero denominator) is NaN.
    """
    metrics = tuple(metrics)
    indicators = confusion_indicators(y_true, y_pred, pos_label)

    def statistic(weights: np.ndarray) -> np.ndarray:
        return binary_metrics_from_counts(weights @ indicators, metrics)

    estimate, dist, used_seed = bootstrap_weighted(
        statistic, len(indicators), n_resamples, confidence, seed, groups,
        chunk_size, show_progress)

    results = {}
    for j, name in enumerate(metrics):
        lo, hi, n_valid = percentile_interval(dist[:, j], confidence)
        results[name] = BootstrapResult(
            estimate=float(estimate[j]), ci_lower=lo, ci_upper=hi,
            distribution=dist[:, j], n_valid=n_valid, n_resamples=n_resamples,
            confidence=confidence, seed=used_seed)
    return results


def bootstrap_binary_metric(y_true, y_pred, metric: str = "f1", **kwargs) -> BootstrapResult:
    """Single-metric convenience wrapper around :func:`bootstrap_binary_metrics`."""
    return bootstrap_binary_metrics(y_true, y_pred, metrics=(metric,), **kwargs)[metric]
