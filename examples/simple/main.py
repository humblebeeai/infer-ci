#!/usr/bin/env python
"""
Simple Binary Classification Example with Confidence Intervals

This example demonstrates how to compute classification metrics with
confidence intervals using the infer-ci package.
"""

import numpy as np
from infer_ci import MetricEvaluator, accuracy_score

def main():
    print("Binary Classification with Confidence Intervals")
    print("=" * 50)

    # Sample binary classification data
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

    print(f"\nDataset size: {len(y_true)} samples")
    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")

    # Method 1: Using MetricEvaluator (Recommended)
    print("\n" + "=" * 50)
    print("Method 1: Using MetricEvaluator")
    print("=" * 50)

    evaluator = MetricEvaluator()

    # Compute accuracy with Wilson score interval
    acc, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric='accuracy',
        method='wilson'
    )
    print(f"\nAccuracy: {acc:.3f}")
    print(f"95% CI (Wilson): [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Compute F1 score with bootstrap
    f1, ci_f1 = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric='f1',
        method='bootstrap_bca',
        n_resamples=1000
    )
    print(f"\nF1 Score: {f1:.3f}")
    print(f"95% CI (Bootstrap BCA): [{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]")

    # Method 2: Direct metric function call
    print("\n" + "=" * 50)
    print("Method 2: Direct Function Call")
    print("=" * 50)

    acc2, ci2 = accuracy_score(y_true, y_pred, method='wilson')
    print(f"\nAccuracy: {acc2:.3f}")
    print(f"95% CI: [{ci2[0]:.3f}, {ci2[1]:.3f}]")

    # Compute multiple metrics at once
    print("\n" + "=" * 50)
    print("Computing Multiple Metrics")
    print("=" * 50)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        value, ci = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            task='classification',
            metric=metric,
            method='wilson'
        )
        print(f"{metric.capitalize():12s}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

if __name__ == "__main__":
    main()
