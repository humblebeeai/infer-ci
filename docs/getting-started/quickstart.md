
# Quick Start

Get started with infer-ci in 5 minutes.

## Installation

```bash
pip install infer-ci
```

## Your First Evaluation

### Classification

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()
accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="classification",
    metric="accuracy",
    method="wilson"
)

print(f"Accuracy: {accuracy:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**Output:**
```
Accuracy: 0.900
95% CI: [0.597, 0.983]
```

### Regression

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5])

evaluator = MetricEvaluator()
mae, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="regression",
    metric="mae",
    method="bootstrap_bca",
    n_resamples=2000
)

print(f"MAE: {mae:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**Output:**
```
MAE: 0.600
95% CI: [0.200, 1.100]
```

## Direct Function Calls

```python
from infer_ci import accuracy_score, mae
import numpy as np

# Classification
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

acc, ci = accuracy_score(y_true, y_pred, method="wilson")
print(f"Accuracy: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

# Regression
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae_val, ci = mae(y_true, y_pred)
print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Common Workflows

### 1. Evaluating Multiple Metrics

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()
metrics = ["accuracy", "precision", "recall", "f1"]

for metric in metrics:
    value, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task="classification",
        metric=metric,
        method="wilson"
    )
    print(f"{metric.capitalize():12s}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**Output:**
```
Accuracy    : 0.900 [0.597, 0.983]
Precision   : 0.857 [0.485, 0.981]
Recall      : 1.000 [0.661, 1.000]
F1          : 0.923 [0.640, 0.989]
```

### 2. Comparing CI Methods

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

methods = ["wilson", "normal", "bootstrap_bca"]

for method in methods:
    acc, ci = accuracy_score(y_true, y_pred, method=method, n_resamples=2000)
    width = ci[1] - ci[0]
    print(f"{method:15s}: [{ci[0]:.3f}, {ci[1]:.3f}] (width: {width:.3f})")
```

**Output:**
```
wilson         : [0.519, 0.938] (width: 0.419)
normal         : [0.596, 1.004] (width: 0.408)
bootstrap_bca  : [0.533, 0.933] (width: 0.400)
```

### 3. Multi-Class Classification

```python
from infer_ci import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)
```

**Output:**
```
    Class  Precision  Recall  F1-Score    Precision CI       Recall CI     F1-Score CI  Support
0  Class 0      0.600   1.000     0.750  (0.231, 0.882)    (0.439, 1.0)  (0.408, 1.092)        3
1  Class 1      0.889   1.000     0.941   (0.565, 0.98)    (0.676, 1.0)  (0.796, 1.086)        8
2  Class 2      1.000   0.667     0.800     (0.61, 1.0)  (0.354, 0.879)  (0.562, 1.038)        9
3    micro      0.850   0.850     0.850  (0.694, 1.006)  (0.694, 1.006)  (0.694, 1.006)       20
4    macro      0.830   0.889     0.830  (0.702, 0.958)  (0.775, 1.002)  (0.548, 1.113)       20
```

## Available Metrics

### Classification Metrics

```python
from infer_ci import MetricEvaluator

evaluator = MetricEvaluator()
print(evaluator.get_available_metrics("classification"))
```

- `accuracy` — Overall accuracy
- `precision` — Positive predictive value
- `recall` — True positive rate
- `f1` — F1 score
- `specificity` — True negative rate
- `npv` — Negative predictive value
- `fpr` — False positive rate
- `auc` — ROC AUC score

### Regression Metrics

```python
print(evaluator.get_available_metrics("regression"))
```

- `mae` — Mean absolute error
- `mse` — Mean squared error
- `rmse` — Root mean squared error
- `r2` — R² score
- `mape` — Mean absolute percentage error
- `iou` — Intersection over Union

### Detection Metrics

```python
print(evaluator.get_available_metrics("detection"))
```

- `map` — Mean average precision at IoU 0.5:0.95
- `map50` — Mean average precision at IoU 0.5
- `precision` — Detection precision
- `recall` — Detection recall

## Available CI Methods

### Classification

```python
print(evaluator.get_available_methods("classification"))
```

**Analytical methods:**
- `wilson` — Wilson score interval (recommended)
- `normal` — Normal approximation
- `agresti_coull` — Agresti-Coull interval
- `beta` — Clopper-Pearson interval
- `jeffreys` — Jeffreys interval

**Bootstrap methods:**
- `bootstrap_bca` — BCa bootstrap (recommended)
- `bootstrap_percentile` — Percentile bootstrap
- `bootstrap_basic` — Basic bootstrap

**Other methods:**
- `jackknife` — Jackknife resampling
- `takahashi` — For multi-class F1, precision, and recall

### Regression

```python
print(evaluator.get_available_methods("regression"))
```

- `bootstrap_bca` — BCa bootstrap (default)
- `bootstrap_percentile` — Percentile bootstrap
- `jackknife` — Jackknife resampling

## Real-World Example

### Evaluating a Scikit-Learn Model

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from infer_ci import MetricEvaluator

iris = load_iris()
X, y = iris.data, iris.target

# Binary classification (class 0 vs. the rest)
y_binary = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

evaluator = MetricEvaluator()

for metric in ["accuracy", "precision", "recall", "f1"]:
    value, ci = evaluator.evaluate(
        y_true=y_test,
        y_pred=y_pred,
        task="classification",
        metric=metric,
        method="wilson"
    )
    print(f"{metric.capitalize():12s}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Next Steps

- [Classification Metrics Guide](classification.md)
- [Regression Metrics Guide](regression.md)
- [Object Detection Guide](../object-detection-metrics-usage.md)
- [CI Methods Explained](ci-methods.md)
- [API Reference](../api-docs/index.md)

## Common Parameters

- `method` — CI calculation method (default: `bootstrap_bca`)
- `confidence_interval` — Confidence level (default: `0.95`)
- `n_resamples` — Number of bootstrap iterations (default: `2000`)
- `random_state` — Random seed for reproducibility
- `compute_ci` — Whether to compute a confidence interval (default: `True`)
- `plot` — Generate a visualization (default: `False`, bootstrap only)

## Tips

1. Use the Wilson method for binary classification metrics.
2. Use Bootstrap BCA for regression metrics.
3. Set `random_state` for reproducible results.
4. Increase `n_resamples` (e.g., `5000`) for more stable confidence intervals.
5. Use `plot=True` with bootstrap methods to visualize the distribution.

## Getting Help

- [Documentation](https://infer.humblebee.ai)
- [GitHub Discussions](https://github.com/humblebeeai/infer-ci/discussions)
- [Report an Issue](https://github.com/humblebeeai/infer-ci/issues)
