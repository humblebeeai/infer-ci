
# Quick Start

Get up and running with infer-ci in 5 minutes.

## Installation

Install infer-ci using pip:

```bash
pip install infer-ci
```

## Your First Evaluation

### Classification Example

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()

accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)

print(f"Accuracy: {accuracy:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Regression Example

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5])

evaluator = MetricEvaluator()

mae, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='bootstrap_bca',
    n_resamples=2000
)

print(f"MAE: {mae:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Direct Function Calls

```python
from infer_ci import accuracy_score, mae
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

acc, ci = accuracy_score(y_true, y_pred, method='wilson')
print(f"Accuracy: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae_val, ci = mae(y_true, y_pred)
print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Evaluating Multiple Metrics

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    value, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric.capitalize()}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Comparing Confidence Interval Methods

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

methods = ['wilson', 'normal', 'bootstrap_bca']

for method in methods:
    acc, ci = accuracy_score(
        y_true, y_pred, method=method, n_resamples=2000
    )
    width = ci[1] - ci[0]
    print(f"{method:15s}: [{ci[0]:.3f}, {ci[1]:.3f}] (width: {width:.3f})")
```

## Multi-Class Classification

```python
from infer_ci import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)
```

## Available Metrics

### Classification Metrics

- `accuracy` - Overall accuracy
- `precision` - Positive predictive value
- `recall` - True positive rate
- `f1` - F1 score
- `specificity` - True negative rate
- `npv` - Negative predictive value
- `fpr` - False positive rate
- `auc` - Area under the ROC curve

### Regression Metrics

- `mae` - Mean absolute error
- `mse` - Mean squared error
- `rmse` - Root mean squared error
- `r2` - R-squared score
- `mape` - Mean absolute percentage error
- `iou` - Intersection over Union

### Object Detection Metrics

- `map` - Mean Average Precision at IoU thresholds 0.5:0.95
- `map50` - Mean Average Precision at IoU 0.5
- `precision` - Detection precision
- `recall` - Detection recall

## Available Confidence Interval Methods

### Classification

- `wilson` - Wilson score interval
- `normal` - Normal approximation
- `agresti_coull` - Agresti-Coull interval
- `beta` - Clopper-Pearson interval
- `jeffreys` - Jeffreys interval
- `bootstrap_bca` - Bias-corrected and accelerated bootstrap
- `bootstrap_percentile` - Percentile bootstrap
- `bootstrap_basic` - Basic bootstrap interval
- `jackknife` - Jackknife resampling
- `takahashi` - Method for multi-class precision, recall, and F1

### Regression

- `bootstrap_bca` - Bootstrap BCa interval
- `bootstrap_percentile` - Bootstrap percentile interval
- `jackknife` - Jackknife resampling

## Real-World Example

### Evaluating a Scikit-Learn Model

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from infer_ci import MetricEvaluator

iris = load_iris()
X, y = iris.data, iris.target

# Binary classification: class 0 vs. all other classes
y_binary = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

evaluator = MetricEvaluator()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    value, ci = evaluator.evaluate(
        y_true=y_test,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric.capitalize()}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Next Steps

- [Classification Metrics Guide](classification.md)
- [Regression Metrics Guide](regression.md)
- [Object Detection Guide](../object-detection-metrics-usage.md)
- [Confidence Interval Methods](ci-methods.md)
- [API Reference](../api-docs/index.md)

## Common Parameters

- `method` - Confidence interval calculation method
- `confidence_interval` - Confidence level (default: `0.95`)
- `n_resamples` - Number of bootstrap iterations (default: `2000`)
- `random_state` - Random seed for reproducibility
- `compute_ci` - Whether to compute confidence intervals (default: `True`)
- `plot` - Whether to generate a visualization (default: `False`)

## Tips

1. Use `wilson` for binary classification metrics.
2. Use `bootstrap_bca` for bootstrap confidence intervals.
3. Set `random_state` for reproducible results.
4. Increase `n_resamples` for more stable bootstrap estimates.
5. Use `plot=True` with supported bootstrap methods to visualize the distribution.

## Getting Help

- [Full Documentation](https://infer.humblebee.ai)
- [GitHub Discussions](https://github.com/humblebeeai/infer-ci/discussions)
- [Report Issues](https://github.com/humblebeeai/infer-ci/issues)
