"""Time the vectorized bootstrap engine against the scipy-based ``bootstrap_ci``.

Usage:
    python benchmarks/bench_bootstrap_engine.py [--n 10000] [--resamples 9999]

The scipy path calls a Python function once per resample, so it is timed on
fewer resamples (``--scipy-resamples``) and the reported time is scaled up.
"""

import argparse
import platform
import time

import numpy as np
from sklearn.metrics import f1_score

from infer_ci import bootstrap_ci
from infer_ci.bootstrap_engine import bootstrap_binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--resamples", type=int, default=9999)
    parser.add_argument("--scipy-resamples", type=int, default=500)
    parser.add_argument("--groups", type=int, default=200,
                        help="number of groups for the grouped run (0 to skip)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    y = (rng.random(args.n) < 0.1).astype(int)
    p = np.where(rng.random(args.n) < 0.85, y, 1 - y)
    print(f"{platform.machine()} | Python {platform.python_version()} | numpy {np.__version__}")
    print(f"n={args.n}, resamples={args.resamples}")

    start = time.perf_counter()
    bootstrap_binary_metrics(y, p, n_resamples=args.resamples, seed=0)
    print(f"engine, 4 metrics:            {time.perf_counter() - start:7.2f} s")

    if args.groups:
        groups = rng.integers(0, args.groups, size=args.n)
        start = time.perf_counter()
        bootstrap_binary_metrics(y, p, n_resamples=args.resamples, seed=0, groups=groups)
        print(f"engine, 4 metrics, {args.groups} groups: {time.perf_counter() - start:7.2f} s")

    start = time.perf_counter()
    bootstrap_ci(y, p, f1_score, n_resamples=args.scipy_resamples,
                 method="bootstrap_percentile", random_state=0, show_progress=False)
    elapsed = (time.perf_counter() - start) * args.resamples / args.scipy_resamples
    print(f"bootstrap_ci, f1 only (est.): {elapsed:7.2f} s")


if __name__ == "__main__":
    main()
