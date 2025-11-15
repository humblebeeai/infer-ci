# Developer Guide: Adding New Metrics

This guide describes how to add new metrics to the confidence interval library. Follow these steps to contribute a new metric while maintaining consistency with the existing codebase.

## Overview

Adding a new metric requires updates to several files in the repository:
- The appropriate metrics module (`regression_metrics.py` or `binary_metrics.py`)
- `evaluator.py` (imports, docstrings, and mappings)
- `__init__.py` (exports)
- `README.md` (documentation)

---

## Step-by-Step Guide

### Step 1: Define Your Metric Function

Add your metric function to the appropriate module:
- **Regression metrics**: `regression_metrics.py`
- **Classification/Binary metrics**: `binary_metrics.py`

#### Required Function Signature

Your metric function must follow this structure:

```python
def your_metric_name(y_true: List[float],
                     y_pred: List[float],
                     confidence_level: float = 0.95,
                     method: str = 'bootstrap_bca',
                     compute_ci: bool = True,
                     plot: bool = False,
                     **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute [Your Metric Name] and optionally the confidence interval.
    
    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        If true return the confidence interval as well as the metric score, by default True
    plot : bool, optional
        If true create histogram plot for bootstrap methods, by default False

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric score and optionally the confidence interval.
    """
    def your_metric_calculation(y_true, y_pred):
        # Implement your metric calculation here
        # Example: return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        pass
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=your_metric_calculation,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="your_metric_name",
        **kwargs
    )
```

#### Example: Mean Absolute Error (MAE)

```python
def mae(y_true: List[float],
        y_pred: List[float],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        compute_ci: bool = True,
        plot: bool = False,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Mean Absolute Error and optionally the confidence interval.
    
    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        If true return the confidence interval as well as the MAE score, by default True
    plot : bool, optional
        If true create histogram plot for bootstrap methods, by default False

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The MAE score and optionally the confidence interval.
    """
    def mae_metric(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=mae_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="mae",
        **kwargs
    )
```

---

### Step 2: Update `evaluator.py` Imports

Add your new metric to the imports section at the top of `evaluator.py`.

**For regression metrics:**
```python
# Import regression metrics
from .regression_metrics import (
    mae, mse, rmse, r2_score, mape, your_new_metric, regression_conf_methods
)
```

**For classification metrics:**
```python
# Import classification metrics
from .binary_metrics import (
    accuracy_score, ppv_score, tpr_score, your_new_metric
)
```

---

### Step 3: Update `evaluator.py` Class Docstring

Add your metric to the `MetricEvaluator` class docstring so users can see all available metrics:

```python
class MetricEvaluator:
    """
    Unified evaluator for classification and regression metrics with confidence intervals.
    
    This class provides a clean interface to compute various metrics with confidence intervals
    for both classification and regression tasks without needing to import individual metrics.

    Available Metrics:
    ------------------
    Classification:
        - accuracy
        - precision (PPV) 
        - recall (TPR / Sensitivity)
        - specificity (TNR)
        - npv (Negative Predictive Value)
        - fpr (False Positive Rate)
        - f1 (F1 Score, Takahashi method)
        - precision_takahashi (Takahashi method)
        - recall_takahashi (Takahashi method)
        - auc (ROC AUC Score)
        - your_new_classification_metric (Description)
    
    Regression:
        - mae (Mean Absolute Error)
        - mse (Mean Squared Error)      
        - rmse (Root Mean Squared Error)
        - r2 (Coefficient of Determination)
        - mape (Mean Absolute Percentage Error)
        - your_new_regression_metric (Description)
    """
```

---

### Step 4: Update `evaluator.py` Metric Mappings

Add your metric to the appropriate dictionary in the `__init__` method of `MetricEvaluator`:

**For regression metrics:**

```python
def __init__(self):
    """Initialize the MetricEvaluator with available metrics and methods."""
    
    # Regression metrics mapping
    self.regression_metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2_score,
        'mape': mape,
        'your_new_metric': your_new_metric  # Add your metric here
    }
    
    # Expose individual metric functions as instance methods
    self.mae = mae
    self.mse = mse
    self.rmse = rmse
    self.r2_score = r2_score
    self.mape = mape
    self.your_new_metric = your_new_metric  # Add your metric here
```

**For classification metrics:**
```python
def __init__(self):
    """Initialize the MetricEvaluator with available metrics and methods."""
    
    # Classification metrics mapping
    self.classification_metrics = {
        'accuracy': accuracy_score,
        'precision': ppv_score,
        'recall': tpr_score,
        'specificity': tnr_score,
        'npv': npv_score,
        'fpr': fpr_score,
        'f1': f1_score,
        'precision_takahashi': precision_score,
        'recall_takahashi': recall_score,
        'auc': roc_auc_score,
        'your_new_metric': your_new_metric  # Add your metric here
    }
    
    # Expose individual metric functions as instance methods
    self.accuracy_score = accuracy_score
    self.ppv_score = ppv_score
    # ... other metrics ...
    self.your_new_metric = your_new_metric  # Add your metric here
```

---

### Step 5: Update `__init__.py` Imports

Add your metric to the import statements in `__init__.py`:

**For regression metrics:**
```python
from .regression_metrics import mae, \
    mse, \
    rmse, \
    r2_score, \
    mape, \
    adjusted_r2_score, \
    sym_mean_abs_per_error, \
    rmse_log, \
    med_abs_err, \
    huber_loss, \
    exp_var_score, \
    mean_bia_dev, \
    your_new_metric  # Add your metric here
```

**For classification metrics:**
```python
from .binary_metrics import accuracy_score, \
    ppv_score, \
    npv_score, \
    tpr_score, \
    fpr_score, \
    tnr_score, \
    your_new_metric  # Add your metric here
```

---

### Step 6: Update `__init__.py` Exports

Add your metric to the `__all__` list in `__init__.py`:

```python
__all__ = [
    # Main unified interface
    'MetricEvaluator', 
    'evaluate_metric',
    'TaskType',
    
    # Classification metrics
    'accuracy_score', 'ppv_score', 'npv_score', 'tpr_score', 
    'fpr_score', 'tnr_score', 'precision_score', 'recall_score', 
    'f1_score', 'roc_auc_score', 'classification_report_with_ci',
    
    # Regression metrics  
    'mae', 'mse', 'rmse', 'r2_score', 'mape', 'adjusted_r2_score',
    'sym_mean_abs_per_error', 'rmse_log', 'med_abs_err', 'huber_loss',
    'exp_var_score', 'mean_bia_dev', 'your_new_metric',  # Add here
    
    # Utility modules
    'methods', 'utils'
]
```

---

### Step 7: Update `README.md` Documentation

Update the README in two places:

#### 7.1: Available Metrics List

Add your metric to the appropriate section:


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
- `your_new_metric`: Brief description of your metric


**Regression:**
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `r2`: Coefficient of Determination
- `mape`: Mean Absolute Percentage Error
- `your_new_metric`: Brief description of your metric


#### 7.2: Usage Examples

Add a usage example in the appropriate section:

**For regression metrics:**
```python
## Regression metrics
from confidenceinterval import MetricEvaluator

evaluate = MetricEvaluator()

r2, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='r2', plot=True)
mse, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mse', plot=True)
mae, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mae', plot=True)
rmse, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='rmse', plot=True)
mape, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='mape', plot=True)
your_metric, ci = evaluate.evaluate(y_true, y_pred, task='regression', metric='your_new_metric', plot=True)
```

**For classification metrics:**
## Classification metrics
```python
from confidenceinterval import MetricEvaluator

evaluate = MetricEvaluator()

f1, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='f1')
precision, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='precision')
recall, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='recall')
your_metric, ci = evaluate.evaluate(y_true, y_pred, task='classification', metric='your_new_metric')
```

## Checklist

Before submitting your contribution, ensure you have:

- [ ] Defined the metric function in the appropriate module (`regression_metrics.py` or `binary_metrics.py`)
- [ ] Added imports to `evaluator.py`
- [ ] Updated the `MetricEvaluator` class docstring in `evaluator.py`
- [ ] Added the metric to the appropriate dictionary in `evaluator.py.__init__()`
- [ ] Exposed the metric as an instance method in `evaluator.py.__init__()`
- [ ] Added imports to `__init__.py`
- [ ] Added the metric to `__all__` in `__init__.py`
- [ ] Updated the available metrics list in `README.md`
- [ ] Added usage examples to `README.md`
- [ ] Tested your metric with sample data
- [ ] Ensured your function follows the required signature and uses `_compute_metric_with_optional_ci`

---

## Best Practices

1. **Consistent Naming**: Use clear, descriptive names that follow the existing naming conventions
2. **Complete Documentation**: Include comprehensive docstrings with parameter descriptions and return types
3. **Type Hints**: Use proper type hints for all function parameters and return values
4. **Error Handling**: Ensure your metric handles edge cases appropriately
5. **Testing**: Test your metric with various input scenarios before submitting
6. **Code Style**: Follow the existing code style and formatting conventions

---

## Questions?

If you have questions about contributing metrics, please open an issue in the repository.