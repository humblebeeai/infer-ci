# Confidence Intervals for Machine Learning Metrics

A comprehensive Python library for computing confidence intervals along with machine learning metrics including both classification and regression metrics.

## Installation

Clone this repository and install in development mode:

```bash
git clone https://github.com/humblebeeai/evaluation_tool.git
cd evaluation_tool
pip install -e .
```

## Overview

This package computes confidence intervals for a wide range of machine learning evaluation metrics, providing both analytical and bootstrap methods for interval estimation.

**Key Features:**
- **Classification Metrics**: Accuracy, Precision, Recall, Negative Predictive Value, Specifity, False Positive Rate
- **Regression Metrics**: MAE, MSE, RMSE, R², MAPE
- **Multiple Methods**: Wilson, Normal, Agresti-Coull, Beta, Jeffreys, Binom_test, DeLong, Takahashi, Bootstrap (BCA, Percentile, Basic), Jackknife 


## Why Confidence Intervals Matter

Confidence intervals provide lower and upper bounds on your metrics, helping you understand the uncertainty in your measurements. They are influenced by sample size and metric sensitivity to data variations.

Wide confidence intervals suggest that high metric values might be due to chance, while narrow intervals indicate more reliable performance estimates. This is particularly important when making model selection decisions or comparing different approaches.


## Quick Start

### Method 1: Unified Evaluator Interface (Recommended)

#### To get started
```python
from confidenceinterval import MetricEvaluator

# Initialize evaluator
evaluator = MetricEvaluator()
```
#### Classification example
```python
# Classification examples
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0] # Ground Truth
y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1] # Predicted

# Single metric evaluation
accuracy, ci = evaluator.evaluate(
    y_true=y_true, 
    y_pred=y_pred, 
    task='classification', 
    metric='accuracy',
    method='wilson'  # analytical method
)
print(f"Accuracy: {accuracy:.3f}, CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Multiple metrics at once
cls_results = evaluator.evaluate_multiple(
    y_true=y_true_cls,
    y_pred=y_pred_cls,
    task='classification',
    metrics=['accuracy', 'precision', 'recall', 'f1'],
    method='bootstrap_bca'
)

for metric, (value, ci) in cls_results.items():
    print(f"{metric}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```
#### Regression examples
```python
# Regression examples  
y_true = [1.0, 2.0, 3.0, 4.0, 5.0] # Ground Truth
y_pred = [1.1, 2.1, 2.9, 4.1, 4.9] # Predicted

# Single regression metric
mae_val, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='jackknife'
)
print(f"MAE: {mae_val:.3f}, CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Multiple regression metrics
reg_results = evaluator.evaluate_multiple(
    y_true=y_true_reg,
    y_pred=y_pred_reg,
    task='regression', 
    metrics=['mae', 'rmse', 'r2'],
    method='bootstrap_bca', # default if not specified
    n_resamples=5000
)

for metric, (value, ci) in reg_results.items():
    print(f"{metric}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

#### Direct access to individual metric functions

```python
accuracy_direct, ci = evaluator.accuracy_score(y_true_cls, y_pred_cls, method='wilson')
mae_direct, ci = evaluator.mae(y_true_reg, y_pred_reg, method='jackknife')
```

### Method 2: Direct Function Imports
```python
# Import specific metrics directly
from confidenceinterval import accuracy_score, ppv_score, tpr_score, f1_score
from confidenceinterval import mae, mse, rmse, r2_score

# Use directly
accuracy, ci = accuracy_score(y_true, y_pred, confidence_level=0.95, method='wilson')
r2, ci = r2_score(y_true, y_pred, confidence_level=0.95, method='bootstrap_bca')
```

## Available Metrics

You can check available metrics and methods programmatically:

```python
from confidenceinterval import MetricEvaluator

evaluator = MetricEvaluator()

# Get available metrics for each task
classification_metrics = evaluator.get_available_metrics('classification')
regression_metrics = evaluator.get_available_metrics('regression')

print("Classification metrics:", classification_metrics)
print("Regression metrics:", regression_metrics)

# Get available methods
classification_methods = evaluator.get_available_methods('classification')
regression_methods = evaluator.get_available_methods('regression')

print("Classification methods:", classification_methods)
print("Regression methods:", regression_methods)

# Print comprehensive info about all available options
evaluator.print_info()

# Access individual metric functions directly from evaluator
evaluator.accuracy_score  # Same as accuracy_score function
evaluator.mae            # Same as mae function
evaluator.f1_score       # Same as f1_score function
```

**Core Metrics Available via MetricEvaluator:**

**Classification:**
- `accuracy`: Overall classification accuracy
- `precision`: Positive Predictive Value (PPV) 
- `recall`: True Positive Rate/Sensitivity (TPR)
- `specificity`: True Negative Rate (TNR)
- `npv`: Negative Predictive Value
- `fpr`: False Positive Rate
- `f1`: F1 Score (supports averaging)
- `precision_takahashi`: Precision using Takahashi method
- `recall_takahashi`: Recall using Takahashi method
- `auc`: ROC AUC Score

**Regression:**
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `r2`: Coefficient of Determination
- `mape`: Mean Absolute Percentage Error

## Supported Methods

### Classification Methods
- **Analytical Methods (Default)**: Wilson, Normal, Agresti-Coull, Beta, Jeffreys, Binom_test
- **Bootstrap Methods**: bootstrap_bca, bootstrap_percentile, bootstrap_basic

### Regression Methods  
- **Bootstrap Methods**: bootstrap_bca, bootstrap_percentile, bootstrap_basic
- **Jackknife Method**: jackknife

Additional bootstrap parameters:
```python
import numpy as np
random_state = np.random.default_rng(42)  # For reproducibility
n_resamples = 9999  # Number of bootstrap samples
```

## Classification Metrics

### F1, Precision, and Recall

```python
from confidenceinterval import MetricEvaluator, precision_score, recall_score, f1_score

# Using unified evaluator (simplest approach)
evaluate = MetricEvaluator()

# Binary F1, Precision, Recall
f1, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='f1')
precision, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='precision')
recall, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='recall')

# Direct function calls with averaging options
binary_f1, ci = f1_score(y_true, y_pred, confidence_level=0.95, average='binary')
macro_f1, ci = f1_score(y_true, y_pred, confidence_level=0.95, average='macro')
micro_f1, ci = f1_score(y_true, y_pred, confidence_level=0.95, average='micro')

# Precision and recall with different averaging
precision_macro, ci = precision_score(y_true, y_pred, average='macro')
recall_micro, ci = recall_score(y_true, y_pred, average='micro')

# Using bootstrap methods
f1_bootstrap, ci = f1_score(y_true, y_pred, method='bootstrap_bca', n_resamples=5000)
```

The analytical computation uses the Takahashi et al. (2022) method for micro averaging, with macro averaging derived using the delta method.

### ROC AUC
```python
from confidenceinterval import roc_auc_score, MetricEvaluator

# Using evaluator
evaluate = MetricEvaluator()
auc, ci = evaluate.evaluate(y_true, y_scores, task='classification', metric='auc')

# Direct function call
auc, ci = roc_auc_score(y_true, y_scores, confidence_level=0.95)
```

Uses a fast implementation of the DeLong method for analytical computation.


### Binary Classification Metrics

```python
from confidenceinterval import MetricEvaluator

# Initialize evaluator  
evaluate = MetricEvaluator()

# Using unified interface (all available metrics)
accuracy, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='accuracy')
precision, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='precision')
recall, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='recall')
specificity, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='specificity')
npv, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='npv')
fpr, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='fpr')
f1, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='f1')
auc, ci = evaluate.evaluate(y_true, y_scores, task='classification', metric='auc')

# Or using direct function calls
from confidenceinterval import accuracy_score, ppv_score, tpr_score, tnr_score

accuracy, ci = accuracy_score(y_true, y_pred, confidence_level=0.95)
ppv, ci = ppv_score(y_true, y_pred, confidence_level=0.95)  # Same as precision
tpr, ci = tpr_score(y_true, y_pred, confidence_level=0.95)  # Same as recall
tnr, ci = tnr_score(y_true, y_pred, confidence_level=0.95)  # Same as specificity

# Different methods
ppv_wilson, ci = ppv_score(y_true, y_pred, method='wilson')  # Default
ppv_bootstrap, ci = ppv_score(y_true, y_pred, method='bootstrap_bca')
```

**Available Binary Metrics via MetricEvaluator:**
- **accuracy**: Overall classification accuracy
- **precision**: Positive Predictive Value (PPV)
- **recall**: True Positive Rate (TPR) / Sensitivity
- **specificity**: True Negative Rate (TNR)
- **npv**: Negative Predictive Value
- **fpr**: False Positive Rate
- **f1**: F1 Score (Takahashi method)
- **auc**: ROC AUC Score

These methods treat the ratio as a binomial proportion. Available analytical methods:
`['wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys', 'binom_test']`

The Wilson interval is used by default as it performs better with smaller samples.

## Classification Report with Confidence Intervals

Generate comprehensive reports with confidence intervals for all metrics:

```python
from confidenceinterval import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)

     Class  Precision  Recall  F1-Score    Precision CI       Recall CI     F1-Score CI  Support
0  Class 0      0.600   1.000     0.750  (0.231, 0.882)    (0.439, 1.0)  (0.408, 1.092)        3
1  Class 1      0.889   1.000     0.941   (0.565, 0.98)    (0.676, 1.0)  (0.796, 1.086)        8
2  Class 2      1.000   0.667     0.800     (0.61, 1.0)  (0.354, 0.879)  (0.562, 1.038)        9
3    micro      0.850   0.850     0.850  (0.694, 1.006)  (0.694, 1.006)  (0.694, 1.006)       20
4    macro      0.830   0.889     0.830  (0.702, 0.958)  (0.775, 1.002)  (0.548, 1.113)       20
```
Custom class labels and formatting options:
```python
numerical_to_label = {
    0: "Cherries",
    1: "Olives", 
    2: "Tangerines"
}

classification_report_with_ci(y_true, y_pred, 
                             round_ndigits=2,
                             numerical_to_label_map=numerical_to_label,
                             binary_method='wilson')
```


## Custom Metrics with Bootstrap

Compute confidence intervals for any metric using bootstrap methods:

```python
from confidenceinterval.bootstrap import bootstrap_ci
import sklearn.metrics
import numpy as np

# Example with balanced accuracy
random_generator = np.random.default_rng(42)
metric_value, ci = bootstrap_ci(
    y_true=y_true,
    y_pred=y_pred,
    metric=sklearn.metrics.balanced_accuracy_score,
    confidence_level=0.95,
    n_resamples=9999,
    method='bootstrap_bca',
    random_state=random_generator
)
```

## Regression Metrics

```python
from confidenceinterval import MetricEvaluator

# Initialize evaluator
evaluate = MetricEvaluator()

# Using unified interface (recommended)
r2, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='r2')
mse, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mse')
mae, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mae')
rmse, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='rmse')
mape, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mape')

print(f"R²: {r2:.3f}, CI: ({ci[0]:.3f}, {ci[1]:.3f})")
print(f"MSE: {mse:.3f}, CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Or using direct function imports
from confidenceinterval import mae, mse, rmse, r2_score, mape

# Standard regression metrics with different methods
mae_value, ci = mae(y_true, y_pred, confidence_level=0.95)
r2_value, ci = r2_score(y_true, y_pred, confidence_level=0.95, method='bootstrap_bca')
mape_value, ci = mape(y_true, y_pred, confidence_level=0.95, method='jackknife')

# Additional regression metrics (direct import required - not in evaluator)
from confidenceinterval import (adjusted_r2_score, sym_mean_abs_per_error, 
                               rmse_log, med_abs_err, huber_loss, exp_var_score)

adj_r2, ci = adjusted_r2_score(num_features=5, y_true=y_true, y_pred=y_pred)
huber_val, ci = huber_loss(y_true, y_pred, delta=1.0, confidence_level=0.95)
```

**Core Regression Metrics (via MetricEvaluator):**
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

**Additional Regression Metrics (direct import only):**
- **Adjusted R²**: Adjusted Coefficient of Determination
- **SMAPE**: Symmetric Mean Absolute Percentage Error  
- **RMSLE**: Root Mean Squared Logarithmic Error
- **Median AE**: Median Absolute Error
- **Huber Loss**: Robust loss function with delta parameter
- **Explained Variance**: Explained variance score
- **Mean Bias**: Mean bias deviation

## References

This library implements methods from several key research papers:

**Key References:**
- **Binomial Confidence Intervals**: Statsmodels package implementation
- **DeLong Method**: Fast ROC AUC comparison implementation (Yandex Data School)  
- **F1 Score Confidence Intervals**: Takahashi et al. (2022) "Confidence interval for micro-averaged F1 and macro-averaged F1 scores"
- **Bootstrap Methods**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"

