# mae

Compute Mean Absolute Error (MAE) with confidence intervals.

::: infer_ci.regression_metrics.mae
    options:
      show_root_heading: true
      show_source: true

## Overview

Mean Absolute Error (MAE) measures the average magnitude of errors in a set of predictions, without considering their direction. It's the average of the absolute differences between predictions and actual observations.

## Mathematical Definition

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $n$ = number of samples

## Examples

### Basic Usage

```python
from infer_ci import mae
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Using bootstrap BCA method (default)
mae_value, ci = mae(y_true, y_pred)

print(f"MAE: {mae_value:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

Output:
```
MAE: 0.500
95% CI: [0.125, 0.875]
```

### Comparing CI Methods

```python
from infer_ci import mae
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0, 1.0, 4.0, 3.5])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5, 1.5, 3.8, 3.2])

methods = ['bootstrap_bca', 'bootstrap_percentile', 'jackknife']

for method in methods:
    mae_val, ci = mae(y_true, y_pred, method=method, n_resamples=2000)
    print(f"{method:22s}: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

Output:
```
bootstrap_bca         : 0.438 [0.200, 0.688]
bootstrap_percentile  : 0.438 [0.219, 0.700]
jackknife             : 0.438 [0.182, 0.693]
```

### With MetricEvaluator

```python
from infer_ci import MetricEvaluator
import numpy as np

evaluator = MetricEvaluator()

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae_val, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='bootstrap_bca',
    n_resamples=2000
)

print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Real-World Example: Linear Regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from infer_ci import mae
import numpy as np

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate with confidence interval
mae_val, ci = mae(y_test, y_pred, method='bootstrap_bca', n_resamples=2000, random_state=42)

print(f"Mean Absolute Error: {mae_val:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"CI Width: {ci[1] - ci[0]:.3f}")
```

### With Visualization

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.random.normal(10, 2, 100)
y_pred = y_true + np.random.normal(0, 1, 100)

evaluator = MetricEvaluator()

mae_val, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='bootstrap_bca',
    n_resamples=2000,
    plot=True  # Generate histogram of bootstrap distribution
)

print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Parameters

- **y_true** (*array-like*): Ground truth continuous values
- **y_pred** (*array-like*): Predicted continuous values
- **method** (*str, optional*): CI calculation method. Default: 'bootstrap_bca'
  - Bootstrap: 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'
  - Other: 'jackknife'
- **confidence_interval** (*float, optional*): Confidence level (0 < CI < 1). Default: 0.95
- **n_resamples** (*int, optional*): Number of bootstrap resamples. Default: 2000
- **compute_ci** (*bool, optional*): Whether to compute CI. Default: True
- **random_state** (*int, optional*): Random seed for reproducibility

## Returns

- **mae_value** (*float*): The computed MAE value (≥ 0)
- **ci** (*tuple*): Confidence interval (lower_bound, upper_bound)

## Interpretation

### Scale
- **Lower is better**: MAE = 0 indicates perfect predictions
- **Same units as target**: MAE is in the same units as the target variable, making it easy to interpret
- **Range**: [0, ∞)

### Comparing Models

```python
from infer_ci import mae
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0, 1.0, 4.0, 3.5])

# Model 1: Simple predictor
y_pred1 = np.array([2.5, 0.0, 2.0, 8.0, 4.5, 1.5, 3.8, 3.2])

# Model 2: Better predictor
y_pred2 = np.array([3.1, -0.4, 2.1, 7.1, 4.9, 1.1, 4.1, 3.4])

mae1, ci1 = mae(y_true, y_pred1, random_state=42)
mae2, ci2 = mae(y_true, y_pred2, random_state=42)

print(f"Model 1 MAE: {mae1:.3f} [{ci1[0]:.3f}, {ci1[1]:.3f}]")
print(f"Model 2 MAE: {mae2:.3f} [{ci2[0]:.3f}, {ci2[1]:.3f}]")

# Check if CIs overlap
if ci2[1] < ci1[0]:
    print("Model 2 is significantly better!")
elif ci1[1] < ci2[0]:
    print("Model 1 is significantly better!")
else:
    print("No significant difference between models")
```

## Advantages

✅ **Pros:**
- Easy to understand and interpret
- Same units as the target variable
- Less sensitive to outliers than MSE
- All errors weighted equally

## Disadvantages

❌ **Cons:**
- Doesn't indicate direction of errors
- Not differentiable at zero (optimization issues)
- Doesn't penalize large errors as heavily as MSE

## When to Use MAE

### Use MAE when:
- You want errors in original units
- Outliers should not dominate the metric
- All error magnitudes are equally important
- Interpretability is crucial

### Consider alternatives when:
- Large errors should be penalized more → use [MSE](mse.md) or [RMSE](rmse.md)
- You need relative errors → use [MAPE](mape.md)
- You want to explain variance → use [R²](r2_score.md)

## Comparison with Other Metrics

| Metric | Unit | Outlier Sensitivity | Differentiable |
|--------|------|-------------------|----------------|
| MAE    | Original | Low | No |
| MSE    | Squared | High | Yes |
| RMSE   | Original | High | Yes |
| MAPE   | Percentage | Medium | No |

## Edge Cases

### Perfect Predictions

```python
from infer_ci import mae
import numpy as np

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1, 2, 3, 4, 5])

mae_val, ci = mae(y_true, y_pred)

print(f"MAE: {mae_val:.3f}")  # 0.000
print(f"CI: [{ci[0]:.3f}, {ci[1]:.3f}]")  # [0.000, 0.000]
```

### Single Sample

```python
from infer_ci import mae
import numpy as np

y_true = np.array([5.0])
y_pred = np.array([4.0])

mae_val, ci = mae(y_true, y_pred, method='bootstrap_bca')

print(f"MAE: {mae_val:.3f}")  # 1.000
# Note: CI will be very wide due to single sample
```

## See Also

- [mse](mse.md) - Mean Squared Error
- [rmse](rmse.md) - Root Mean Squared Error
- [mape](mape.md) - Mean Absolute Percentage Error
- [r2_score](r2_score.md) - Coefficient of Determination
- [MetricEvaluator](../MetricEvaluator.md) - Unified evaluation interface

## References

- Willmott, C. J., & Matsuura, K. (2005). "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance". Climate Research.
- Chai, T., & Draxler, R. R. (2014). "Root mean square error (RMSE) or mean absolute error (MAE)?". Geoscientific Model Development.
