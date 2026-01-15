# accuracy_score

Compute classification accuracy with confidence intervals.

::: infer_ci.binary_metrics.accuracy_score
    options:
      show_root_heading: true
      show_source: true

## Overview

Accuracy is the fraction of predictions that match the true labels. It's the most intuitive performance measure for classification but can be misleading for imbalanced datasets.

## Mathematical Definition

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

## Examples

### Basic Usage

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

# Using Wilson score interval (recommended)
acc, ci = accuracy_score(y_true, y_pred, method='wilson')

print(f"Accuracy: {acc:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

Output:
```
Accuracy: 0.900
95% CI: [0.597, 0.983]
```

### Comparing Different CI Methods

```python
import numpy as np
from infer_ci import accuracy_score

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

methods = ['wilson', 'normal', 'agresti_coull', 'bootstrap_bca']

for method in methods:
    acc, ci = accuracy_score(y_true, y_pred, method=method, n_resamples=2000)
    print(f"{method:20s}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

Output:
```
wilson              : 0.800 [0.519, 0.938]
normal              : 0.800 [0.596, 1.004]
agresti_coull       : 0.800 [0.529, 0.933]
bootstrap_bca       : 0.800 [0.533, 0.933]
```

### With MetricEvaluator

```python
from infer_ci import MetricEvaluator
import numpy as np

evaluator = MetricEvaluator()

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

acc, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)

print(f"Accuracy: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Without Computing CI

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

# Just get the point estimate
acc, _ = accuracy_score(y_true, y_pred, compute_ci=False)

print(f"Accuracy: {acc:.3f}")
```

## Parameters

- **y_true** (*array-like*): Ground truth binary labels (0 or 1)
- **y_pred** (*array-like*): Predicted binary labels (0 or 1)
- **method** (*str, optional*): CI calculation method. Default: 'bootstrap_bca'
  - Analytical: 'wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys', 'binom_test'
  - Bootstrap: 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'
  - Other: 'jackknife'
- **confidence_interval** (*float, optional*): Confidence level (0 < CI < 1). Default: 0.95
- **n_resamples** (*int, optional*): Number of bootstrap resamples. Default: 2000
- **compute_ci** (*bool, optional*): Whether to compute CI. Default: True
- **random_state** (*int, optional*): Random seed for reproducibility

## Returns

- **accuracy** (*float*): The computed accuracy value (between 0 and 1)
- **ci** (*tuple*): Confidence interval (lower_bound, upper_bound)

## Notes

### When to Use Accuracy

✅ **Good for:**
- Balanced datasets
- Binary or multi-class problems where all classes are equally important
- Quick model comparison

❌ **Not recommended for:**
- Imbalanced datasets (use F1, precision, recall instead)
- Medical diagnosis (prefer sensitivity/specificity)
- When false positives and false negatives have different costs

### Method Recommendations

For accuracy with binary classification:

1. **Wilson score** (recommended): Best coverage properties, handles edge cases well
2. **Agresti-Coull**: Good alternative, similar performance to Wilson
3. **Bootstrap BCA**: Use when you want a non-parametric approach
4. **Normal approximation**: Simple but may give intervals outside [0, 1]

## Edge Cases

### Perfect Accuracy

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([1, 1, 1, 1, 1])
y_pred = np.array([1, 1, 1, 1, 1])

acc, ci = accuracy_score(y_true, y_pred, method='wilson')

print(f"Accuracy: {acc:.3f}")  # 1.000
print(f"CI: [{ci[0]:.3f}, {ci[1]:.3f}]")  # [0.566, 1.000]
```

Note: Even with perfect accuracy, the CI lower bound is not 1.0 due to sampling uncertainty.

### Small Sample Size

```python
from infer_ci import accuracy_score
import numpy as np

# Very small sample
y_true = np.array([0, 1, 1])
y_pred = np.array([0, 1, 1])

acc, ci = accuracy_score(y_true, y_pred, method='wilson')

print(f"Accuracy: {acc:.3f}")  # 1.000
print(f"CI: [{ci[0]:.3f}, {ci[1]:.3f}]")  # Very wide interval
```

## See Also

- [precision_score](precision_score.md) - Positive Predictive Value
- [recall_score](recall_score.md) - True Positive Rate
- [f1_score](f1_score.md) - Harmonic mean of precision and recall
- [MetricEvaluator](../MetricEvaluator.md) - Unified evaluation interface

## References

- Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference". Journal of the American Statistical Association.
- Agresti, A., & Coull, B. A. (1998). "Approximate is better than 'exact' for interval estimation of binomial proportions". The American Statistician.
