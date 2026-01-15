# API Reference

Welcome to the infer-ci API reference documentation. This section provides detailed information about all available metrics, methods, and utilities.

## Overview

The infer-ci package provides a unified interface for computing machine learning evaluation metrics with confidence intervals. All metrics support multiple methods for calculating confidence intervals, including bootstrap, analytical, and jackknife methods.

## Core Interface

- **[MetricEvaluator](MetricEvaluator.md)**: Unified interface for evaluating all metrics

## Metrics by Category

### Classification Metrics

Metrics for evaluating classification models:

- **[accuracy_score](classification/accuracy_score.md)**: Overall classification accuracy
- **[precision_score](classification/precision_score.md)**: Positive Predictive Value (PPV)
- **[recall_score](classification/recall_score.md)**: True Positive Rate/Sensitivity (TPR)
- **[f1_score](classification/f1_score.md)**: F1 Score (harmonic mean of precision and recall)
- **[ppv_score](classification/ppv_score.md)**: Positive Predictive Value
- **[tpr_score](classification/tpr_score.md)**: True Positive Rate
- **[tnr_score](classification/tnr_score.md)**: True Negative Rate/Specificity
- **[npv_score](classification/npv_score.md)**: Negative Predictive Value
- **[fpr_score](classification/fpr_score.md)**: False Positive Rate
- **[roc_auc_score](classification/roc_auc_score.md)**: ROC AUC Score

### Regression Metrics

Metrics for evaluating regression models:

- **[mae](regression/mae.md)**: Mean Absolute Error
- **[mse](regression/mse.md)**: Mean Squared Error
- **[rmse](regression/rmse.md)**: Root Mean Squared Error
- **[r2_score](regression/r2_score.md)**: Coefficient of Determination (R²)
- **[mape](regression/mape.md)**: Mean Absolute Percentage Error
- **[iou](regression/iou.md)**: Intersection over Union

### Object Detection Metrics

Metrics for evaluating object detection models:

- **[map](detection/map.md)**: mean Average Precision @0.5:0.95
- **[map50](detection/map50.md)**: mean Average Precision @0.5
- **[precision](detection/precision.md)**: Detection Precision
- **[recall](detection/recall.md)**: Detection Recall

## Utilities

- **[classification_report_with_ci](utilities/classification_report.md)**: Generate classification report with confidence intervals
- **[CI Methods](utilities/ci_methods.md)**: Available confidence interval calculation methods

## Quick Start

```python
from infer_ci import MetricEvaluator

# Initialize evaluator
evaluator = MetricEvaluator()

# Evaluate a metric
value, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)

print(f"Accuracy: {value:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Common Parameters

Most metric functions share these common parameters:

- `y_true` (array-like): Ground truth labels/values
- `y_pred` (array-like): Predicted labels/values
- `method` (str): Method for computing CI (default: 'bootstrap_bca')
- `confidence_interval` (float): Confidence level between 0 and 1 (default: 0.95)
- `n_resamples` (int): Number of bootstrap resamples (default: 2000)
- `compute_ci` (bool): Whether to compute confidence interval (default: True)
- `random_state` (int, optional): Random seed for reproducibility

## Available CI Methods

### Bootstrap Methods
- `bootstrap_bca`: Bootstrap Bias-Corrected and Accelerated (recommended)
- `bootstrap_percentile`: Bootstrap Percentile method
- `bootstrap_basic`: Basic Bootstrap method

### Analytical Methods (Classification)
- `wilson`: Wilson score interval
- `normal`: Normal approximation
- `agresti_coull`: Agresti-Coull interval
- `beta`: Clopper-Pearson interval
- `jeffreys`: Jeffreys interval
- `binom_test`: Binomial test

### Other Methods
- `jackknife`: Jackknife resampling method
- `takahashi`: Takahashi method for F1, precision, recall (multi-class)

## Return Values

All metric functions return a tuple:

```python
(metric_value, (lower_bound, upper_bound))
```

- `metric_value` (float): The computed metric value
- `(lower_bound, upper_bound)` (tuple): The confidence interval bounds
