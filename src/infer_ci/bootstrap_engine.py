"""Vectorized bootstrap engine.

The engine resamples *units* (rows, or groups such as persons) with replacement
and records, for every resample, how many times each unit was drawn. Any
statistic that can be computed from weighted counts (confusion-matrix cells,
error counts, pair counts) is then evaluated for a whole chunk of resamples
with one matrix product instead of one Python call per resample.

Memory
------
Resamples are processed in chunks. Unless ``chunk_size`` is given, a chunk
holds at most ``MAX_CHUNK_ELEMENTS`` draws (resamples x units), so peak memory
stays bounded (roughly 100-150 MB) however large the input is. Binary metrics
aggregate the confusion indicators per unit first, so with ``groups`` they work
on a ``(b, n_groups)`` matrix and never build a ``(b, n_rows)`` one.

Reproducibility
---------------
The seed actually used is always returned. With ``seed=None`` a fresh seed
below 2**32 is drawn, so it survives JSON and JavaScript (which parse numbers
as doubles; seeds above 2**53 do not round-trip there). Draws come from
``numpy.random.default_rng(seed)`` (PCG64) and do not depend on the chunk size.
They are identical for the same seed, input, ``n_resamples`` *and numpy
version*: numpy only guarantees ``Generator`` streams within a version
(NEP 19), so record the numpy version alongside the seed.

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
#: Upper bound on resamples x units held in one chunk when chunk_size is automatic.
MAX_CHUNK_ELEMENTS = 4_000_000

SeedLike = Optional[Union[int, np.integer]]

#: Binary metrics computable from confusion counts (tp, fp, fn, tn).
BINARY_METRICS = ("accuracy", "precision", "recall", "f1",
                  "specificity", "npv", "fpr")


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Outcome of a bootstrap run for one statistic.

    ``estimate``, ``ci_lower`` and ``ci_upper`` are ``nan`` when undefined: the
    estimate when the statistic is undefined on the original sample, the
    bounds when no resample produced a valid value. Compared by identity,
    because ``distribution`` is an array.
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
    """Return a concrete integer seed; with ``None``, draw a fresh one below 2**32."""
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def _check_counts(n_resamples: int, chunk_size: Optional[int]) -> None:
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if chunk_size is not None and (not isinstance(chunk_size, (int, np.integer))
                                   or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer or None")


def resolve_chunk_size(chunk_size: Optional[int], width: int) -> int:
    """``chunk_size`` if given, else as many resamples as fit in ``MAX_CHUNK_ELEMENTS``.

    ``width`` is the number of columns one resample occupies (units or rows).
    """
    if chunk_size is not None:
        _check_counts(1, chunk_size)
        return int(chunk_size)
    return max(1, min(DEFAULT_CHUNK_SIZE, MAX_CHUNK_ELEMENTS // max(width, 1)))


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
                          chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield ``(b, n_units)`` arrays of drawn unit indices, chunk by chunk.

    Each row of a chunk is one resample: ``n_units`` units drawn uniformly with
    replacement. Exposed so a reference implementation can replay the exact
    same draws.
    """
    if n_units < 1:
        raise ValueError("need at least one unit to resample")
    _check_counts(n_resamples, chunk_size)
    chunk_size = resolve_chunk_size(chunk_size, n_units)
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


def _resample_counts(n_units: int,
                     n_resamples: int,
                     seed: int,
                     chunk_size: int,
                     show_progress: bool) -> Iterator[np.ndarray]:
    """Yield ``(b, n_units)`` draw counts, optionally with a progress bar."""
    chunks = iter_resample_indices(n_units, n_resamples, seed, chunk_size)
    if show_progress:
        from tqdm import tqdm
        chunks = tqdm(chunks, total=-(-n_resamples // chunk_size), desc="Bootstrap")
    for indices in chunks:
        yield unit_counts(indices, n_units)


def iter_resample_weights(n_rows: int,
                          n_resamples: int,
                          seed: int,
                          groups: Optional[Sequence] = None,
                          chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield ``(b, n_rows)`` float weight matrices, one row per resample.

    Without ``groups`` a row's weight is how many times that row was drawn.
    With ``groups`` the units are the distinct group labels: as many groups as
    exist are drawn with replacement, and every row gets its group's count.
    """
    _check_counts(n_resamples, chunk_size)
    row_unit, n_units = _unit_index(n_rows, groups)
    chunk_size = resolve_chunk_size(chunk_size, max(n_rows, n_units))
    for counts in _resample_counts(n_units, n_resamples, seed, chunk_size, False):
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


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def bootstrap_weighted(statistic: Callable[[np.ndarray], np.ndarray],
                       n_rows: int,
                       n_resamples: int = 9999,
                       confidence: float = 0.95,
                       seed: SeedLike = None,
                       groups: Optional[Sequence] = None,
                       chunk_size: Optional[int] = None,
                       show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """Evaluate ``statistic`` on the original sample and on every resample.

    ``statistic`` receives a ``(b, n_rows)`` weight matrix and returns an array
    of shape ``(b,)`` or ``(b, k)`` (NaN where undefined). The original sample
    is evaluated with all weights equal to 1.

    Returns ``(estimate, distribution, seed)`` where ``estimate`` has shape
    ``(k,)`` (or ``()``), ``distribution`` has shape ``(n_resamples, k)`` (or
    ``(n_resamples,)``) and ``seed`` is the seed actually used.
    """
    _check_confidence(confidence)
    _check_counts(n_resamples, chunk_size)
    used_seed = resolve_seed(seed)
    row_unit, n_units = _unit_index(n_rows, groups)
    chunk_size = resolve_chunk_size(chunk_size, max(n_rows, n_units))
    estimate = np.asarray(statistic(np.ones((1, n_rows))), dtype=np.float64)[0]

    parts = []
    for counts in _resample_counts(n_units, n_resamples, used_seed, chunk_size,
                                   show_progress):
        weights = counts if groups is None else counts[:, row_unit]
        parts.append(np.asarray(statistic(weights.astype(np.float64)), dtype=np.float64))
    return estimate, np.concatenate(parts, axis=0), used_seed


def _check_binary(y_true, y_pred, pos_label) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional and of equal length")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    if len(labels) > 2:
        raise ValueError(f"binary metrics need at most two labels, got {labels}")
    if len(labels) == 2 and not any(label == pos_label for label in labels):
        raise ValueError(f"pos_label={pos_label!r} is not one of the labels {labels}")
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
                             chunk_size: Optional[int] = None,
                             show_progress: bool = False) -> Dict[str, BootstrapResult]:
    """Percentile bootstrap CIs for binary metrics, all from the same resamples.

    Metrics are computed directly from confusion counts with ``pos_label`` as
    the positive class: no averaging, and no small constant added to any
    denominator. A metric that is undefined (zero denominator) is NaN.
    Raises ``ValueError`` if two labels are present and neither is ``pos_label``.
    """
    metrics = tuple(metrics)
    _check_confidence(confidence)
    _check_counts(n_resamples, chunk_size)
    binary_metrics_from_counts(np.zeros((1, 4)), metrics)  # validate names early
    indicators = confusion_indicators(y_true, y_pred, pos_label)
    used_seed = resolve_seed(seed)

    # Aggregate tp/fp/fn/tn per resampling unit, so each resample is just
    # (draw counts per unit) @ (counts per unit).
    row_unit, n_units = _unit_index(len(indicators), groups)
    per_unit = np.zeros((n_units, 4))
    np.add.at(per_unit, row_unit, indicators)
    chunk_size = resolve_chunk_size(chunk_size, n_units)

    estimate = binary_metrics_from_counts(per_unit.sum(axis=0, keepdims=True), metrics)[0]
    dist = np.concatenate([
        binary_metrics_from_counts(counts @ per_unit, metrics)
        for counts in _resample_counts(n_units, n_resamples, used_seed, chunk_size,
                                       show_progress)])

    results = {}
    for j, name in enumerate(metrics):
        lo, hi, n_valid = percentile_interval(dist[:, j], confidence)
        results[name] = BootstrapResult(
            estimate=float(estimate[j]), ci_lower=lo, ci_upper=hi,
            distribution=dist[:, j], n_valid=n_valid, n_resamples=int(n_resamples),
            confidence=confidence, seed=used_seed)
    return results


def bootstrap_binary_metric(y_true, y_pred, metric: str = "f1", **kwargs) -> BootstrapResult:
    """Single-metric convenience wrapper around :func:`bootstrap_binary_metrics`."""
    return bootstrap_binary_metrics(y_true, y_pred, metrics=(metric,), **kwargs)[metric]
