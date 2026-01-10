# MetricEvaluator

The `MetricEvaluator` class provides a unified interface for computing all metrics with confidence intervals.

::: infer_ci.evaluator.MetricEvaluator
    options:
      show_root_heading: true
      show_source: true
      members:
        - evaluate
        - get_available_metrics
        - get_available_methods

## Overview

`MetricEvaluator` is the recommended way to compute metrics in infer-ci. It provides a consistent interface across all metric types (classification, regression, detection) and handles method selection automatically.

## Basic Usage

```python
from infer_ci import MetricEvaluator
import numpy as np

# Initialize evaluator
evaluator = MetricEvaluator()

# Binary classification example
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

# Compute accuracy with Wilson interval
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

## Supported Tasks and Metrics

### Classification Task

```python
# Get all available classification metrics
metrics = evaluator.get_available_metrics('classification')
print(metrics)
# ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv', 'fpr', 'auc', ...]
```

Available metrics:
- `accuracy`: Overall accuracy
- `precision`: Positive Predictive Value
- `recall`: True Positive Rate/Sensitivity
- `f1`: F1 Score (supports binary, macro, micro averaging)
- `specificity`: True Negative Rate
- `npv`: Negative Predictive Value
- `fpr`: False Positive Rate
- `auc`: ROC AUC Score
- `precision_takahashi`: Precision with Takahashi method
- `recall_takahashi`: Recall with Takahashi method

### Regression Task

```python
# Get all available regression metrics
metrics = evaluator.get_available_metrics('regression')
print(metrics)
# ['mae', 'mse', 'rmse', 'r2', 'mape', 'iou']
```

Available metrics:
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `r2`: R² Score
- `mape`: Mean Absolute Percentage Error
- `iou`: Intersection over Union

### Detection Task

```python
# Object detection example
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict(source='images/', verbose=False)

map_value, ci = evaluator.evaluate(
    y_true='dataset_path',  # Path to dataset or ground truth
    y_pred=results,
    task='detection',
    metric='map',
    method='bootstrap_percentile',
    n_resamples=1000
)
```

Available metrics:
- `map`: mean Average Precision @0.5:0.95
- `map50`: mean Average Precision @0.5
- `precision`: Detection precision
- `recall`: Detection recall

## Available CI Methods

Check which methods are available for each task:

```python
# Classification methods
methods = evaluator.get_available_methods('classification')
print(methods)
# ['wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys', 'binom_test',
#  'bootstrap_bca', 'bootstrap_percentile', 'jackknife']

# Regression methods
methods = evaluator.get_available_methods('regression')
print(methods)
# ['bootstrap_bca', 'bootstrap_percentile', 'jackknife']
```

## Advanced Usage

### Comparing Different CI Methods

```python
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

evaluator = MetricEvaluator()

# Compare different methods
methods = ['wilson', 'normal', 'agresti_coull', 'bootstrap_bca']

for method in methods:
    acc, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric='accuracy',
        method=method,
        n_resamples=2000  # Only used by bootstrap
    )
    width = ci[1] - ci[0]
    print(f"{method:20s}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (width: {width:.3f})")
```

Output:
```
wilson              : 0.800 [0.519, 0.938] (width: 0.419)
normal              : 0.800 [0.596, 1.004] (width: 0.408)
agresti_coull       : 0.800 [0.529, 0.933] (width: 0.404)
bootstrap_bca       : 0.800 [0.533, 0.933] (width: 0.400)
```

### Regression with Visualization

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate data
X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate with visualization
evaluator = MetricEvaluator()
r2_value, ci = evaluator.evaluate(
    y_true=y_test,
    y_pred=y_pred,
    task='regression',
    metric='r2',
    method='bootstrap_bca',
    n_resamples=2000,
    plot=True  # Generate histogram of bootstrap samples
)

print(f"R² Score: {r2_value:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Multi-class Classification with Macro/Micro Averaging

```python
y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

evaluator = MetricEvaluator()

# Macro F1 (average across classes)
macro_f1, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='f1',
    method='takahashi',  # Analytical method for multi-class
    average='macro'
)

print(f"Macro F1: {macro_f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

# Micro F1 (aggregate contributions of all classes)
micro_f1, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='f1',
    method='takahashi',
    average='micro'
)

print(f"Micro F1: {micro_f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Parameters

### evaluate() method

**Parameters:**

- **y_true** (*array-like*): Ground truth labels/values
- **y_pred** (*array-like*): Predicted labels/values
- **task** (*str*): Type of task - 'classification', 'regression', or 'detection'
- **metric** (*str*): Name of the metric to compute
- **method** (*str, optional*): CI calculation method (default: 'bootstrap_bca')
- **confidence_interval** (*float, optional*): Confidence level (default: 0.95)
- **n_resamples** (*int, optional*): Bootstrap resamples (default: 2000)
- **random_state** (*int, optional*): Random seed for reproducibility
- **plot** (*bool, optional*): Generate visualization (default: False)
- **\*\*kwargs**: Additional metric-specific parameters

**Returns:**

- **metric_value** (*float*): The computed metric value
- **ci** (*tuple*): Confidence interval (lower_bound, upper_bound)

## Method Selection Guide

### When to use analytical methods (Classification)

- **wilson**: Recommended for most binary classification metrics (accuracy, precision, recall)
- **normal**: Simple but may give intervals outside [0,1] for small samples
- **agresti_coull**: Good alternative to Wilson
- **beta/jeffreys**: Conservative intervals, good for small samples
- **binom_test**: Exact method, computationally intensive

### When to use bootstrap methods

- **bootstrap_bca**: Recommended default, corrects for bias and skewness
- **bootstrap_percentile**: Simpler alternative, good for symmetric distributions
- **bootstrap_basic**: Basic method, less accurate than BCA

### When to use jackknife

- **jackknife**: Good for small samples, less computationally intensive than bootstrap

### When to use Takahashi methods

- **takahashi**: Specifically for multi-class F1, precision, recall with macro/micro averaging

## Error Handling

```python
# Invalid task
try:
    evaluator.evaluate(y_true, y_pred, task='invalid', metric='accuracy')
except ValueError as e:
    print(e)  # "Task must be one of: classification, regression, detection"

# Invalid metric for task
try:
    evaluator.evaluate(y_true, y_pred, task='regression', metric='accuracy')
except ValueError as e:
    print(e)  # "Metric 'accuracy' not available for regression task"

# Invalid method
try:
    evaluator.evaluate(y_true, y_pred, task='regression', metric='mae', method='wilson')
except ValueError as e:
    print(e)  # "Method 'wilson' not available for regression task"
```

## See Also

- [Classification Metrics](classification/accuracy_score.md)
- [Regression Metrics](regression/mae.md)
- [Detection Metrics](detection/map.md)
- [CI Methods Guide](utilities/ci_methods.md)
