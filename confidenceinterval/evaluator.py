"""
Unified evaluator for classification and regression metrics with confidence intervals.

This module provides a single interface for computing metrics with confidence intervals
for both classification and regression tasks.
"""

from typing import List, Union, Tuple, Optional, Dict, Any
import numpy as np
from enum import Enum
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import classification metrics
from .binary_metrics import (
    accuracy_score, ppv_score, npv_score, tnr_score, tpr_score, fpr_score, proportion_conf_methods
)
from .takahashi_methods import precision_score, recall_score, f1_score
from .auc import roc_auc_score

# Import regression metrics
from .regression_metrics import (
    mae, mse, rmse, r2_score, mape, regression_conf_methods
)


class TaskType(Enum):
    """Enumeration for task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


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
    Regression:
        - mae (Mean Absolute Error)
        - mse (Mean Squared Error)      
        - rmse (Root Mean Squared Error)
        - r2 (Coefficient of Determination)
        - mape (Mean Absolute Percentage Error)
    
    Example:
    --------
    >>> evaluator = MetricEvaluator()
    >>> 
    >>> # For regression
    >>> result = evaluator.evaluate(
    ...     y_true=[1, 2, 3, 4, 5], 
    ...     y_pred=[1.1, 2.2, 2.8, 4.1, 4.9],
    ...     task='regression',
    ...     metric='mae',
    ...     method='bootstrap_bca',
    ...     plot=True
    ... ) 
    ... # Specific metric function can also be called directly:  
    >>> mae_val, ci = evaluator.mae(
    ...     y_true=[1, 2, 3, 4, 5], 
    ...     y_pred=[1.1, 2.2, 2.8, 4.1, 4.9],
    ...     confidence_level=0.95,
    ...     method='jackknife'
    ... )
    >>> 
    >>> # For classification  
    >>> result = evaluator.evaluate(
    ...     y_true=[0, 1, 1, 0, 1], 
    ...     y_pred=[0, 1, 0, 0, 1],
    ...     task='classification',
    ...     metric='accuracy',
    ...     method='wilson'
    ... )
    """
    
    def __init__(self):
        """Initialize the MetricEvaluator with available metrics and methods."""
        
        # Classification metrics mapping
        self.classification_metrics = {
            'accuracy': accuracy_score,
            'precision': ppv_score,  # Positive Predictive Value
            'recall': tpr_score,     # True Positive Rate / Sensitivity
            'specificity': tnr_score, # True Negative Rate
            'npv': npv_score,        # Negative Predictive Value
            'fpr': fpr_score,        # False Positive Rate
            'f1': f1_score,          # F1 Score (Takahashi method)
            'precision_takahashi': precision_score,  # Takahashi precision
            'recall_takahashi': recall_score,        # Takahashi recall
            'auc': roc_auc_score     # ROC AUC Score
        }
        
        # Regression metrics mapping
        self.regression_metrics = {
            'mae': mae,           # Mean Absolute Error
            'mse': mse,           # Mean Squared Error
            'rmse': rmse,         # Root Mean Squared Error
            'r2': r2_score,       # Coefficient of Determination
            'mape': mape          # Mean Absolute Percentage Error
        }
        
        # Available methods for each task type
        self.classification_methods = proportion_conf_methods + ['bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic']
        self.regression_methods = regression_conf_methods
        
        # Expose individual metric functions as instance methods
        # Classification metrics
        self.accuracy_score = accuracy_score
        self.ppv_score = ppv_score
        self.tpr_score = tpr_score
        self.tnr_score = tnr_score
        self.fpr_score = fpr_score
        self.npv_score = npv_score
        self.f1_score = f1_score
        self.precision_score = precision_score
        self.recall_score = recall_score
        self.roc_auc_score = roc_auc_score
        
        # Regression metrics
        self.mae = mae
        self.mse = mse
        self.rmse = rmse
        self.r2_score = r2_score
        self.mape = mape
        
    def get_available_metrics(self, task: Union[str, TaskType]) -> List[str]:
        """
        Get list of available metrics for a given task type.
        
        Parameters:
        -----------
        task : str or TaskType
            The task type ('classification' or 'regression')
            
        Returns:
        --------
        List[str]: List of available metric names
        """
        task_str = task.value if isinstance(task, TaskType) else task.lower()
        
        if task_str == 'classification':
            return list(self.classification_metrics.keys())
        elif task_str == 'regression':
            return list(self.regression_metrics.keys())
        else:
            raise ValueError(f"Unknown task type: {task}. Use 'classification' or 'regression'")
    
    def get_available_methods(self, task: Union[str, TaskType]) -> List[str]:
        """
        Get list of available confidence interval methods for a given task type.
        
        Parameters:
        -----------
        task : str or TaskType
            The task type ('classification' or 'regression')
            
        Returns:
        --------
        List[str]: List of available method names
        """
        task_str = task.value if isinstance(task, TaskType) else task.lower()
        
        if task_str == 'classification':
            return self.classification_methods
        elif task_str == 'regression':
            return self.regression_methods
        else:
            raise ValueError(f"Unknown task type: {task}. Use 'classification' or 'regression'")
    
    def evaluate(self, 
                 y_true: List[Union[int, float]], 
                 y_pred: List[Union[int, float]], 
                 task: Union[str, TaskType],
                 metric: str,
                 method: str = 'bootstrap_bca',
                 confidence_level: float = 0.95,
                 compute_ci: bool = True,
                 plot: bool = False,
                 **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
        """
        Evaluate a metric with confidence intervals.
        
        Parameters:
        -----------
        y_true : List[Union[int, float]]
            True labels/values
        y_pred : List[Union[int, float]]  
            Predicted labels/values
        task : str or TaskType
            Task type ('classification' or 'regression')
        metric : str
            Metric name (e.g., 'accuracy', 'mae', 'f1')
        method : str, optional
            Confidence interval method (default: 'bootstrap_bca')
        confidence_level : float, optional
            Confidence level (default: 0.95)
        compute_ci : bool, optional
            Whether to compute confidence intervals (default: True)
        plot : bool, optional
            Whether to create histogram plot for bootstrap methods (default: False)
        **kwargs : dict
            Additional arguments (e.g., n_resamples for bootstrap methods)
            
        Returns:
        --------
        Union[float, Tuple[float, Tuple[float, float]]]
            If compute_ci=False: metric value
            If compute_ci=True: (metric_value, (lower_bound, upper_bound))
            
        Example:
        --------
        >>> evaluator = MetricEvaluator()
        >>> # Regression example
        >>> mae_val, ci = evaluator.evaluate(
        ...     y_true=[1, 2, 3, 4, 5],
        ...     y_pred=[1.1, 2.1, 2.9, 4.1, 4.9], 
        ...     task='regression',
        ...     metric='mae',
        ...     method='jackknife'
        ... )
        >>> print(f"MAE: {mae_val:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        """
        
        # Validate and normalize task type
        task_str = task.value if isinstance(task, TaskType) else task.lower()
        if task_str not in ['classification', 'regression']:
            raise ValueError(f"Unknown task type: {task}. Use 'classification' or 'regression'")
        
        # Get the appropriate metric function and validate
        if task_str == 'classification':
            if metric not in self.classification_metrics:
                available = ', '.join(self.classification_metrics.keys())
                raise ValueError(f"Unknown classification metric: {metric}. Available: {available}")
            metric_func = self.classification_metrics[metric]
            
            # Validate method for classification
            if method not in self.classification_methods:
                available = ', '.join(self.classification_methods)
                raise ValueError(f"Unknown classification method: {method}. Available: {available}")
                
        else:  # regression
            if metric not in self.regression_metrics:
                available = ', '.join(self.regression_metrics.keys())
                raise ValueError(f"Unknown regression metric: {metric}. Available: {available}")
            metric_func = self.regression_metrics[metric]
            
            # Validate method for regression
            if method not in self.regression_methods:
                available = ', '.join(self.regression_methods)
                raise ValueError(f"Unknown regression method: {method}. Available: {available}")
        
        # Set default kwargs for different methods
        if method.startswith('bootstrap') and 'n_resamples' not in kwargs:
            kwargs['n_resamples'] = 9999
        
        # Call the metric function
        try:
            result = metric_func(
                y_true=y_true,
                y_pred=y_pred,
                confidence_level=confidence_level,
                method=method,
                compute_ci=compute_ci,
                plot=plot,
                **kwargs
            )
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error computing {metric} with method {method}: {str(e)}")
    
    def evaluate_multiple(self,
                         y_true: List[Union[int, float]], 
                         y_pred: List[Union[int, float]], 
                         task: Union[str, TaskType],
                         metrics: List[str],
                         method: str = 'bootstrap_bca',
                         confidence_level: float = 0.95,
                         compute_ci: bool = True,
                         plot: bool = False,
                         **kwargs) -> Dict[str, Union[float, Tuple[float, Tuple[float, float]]]]:
        """
        Evaluate multiple metrics at once.
        
        Parameters:
        -----------
        y_true, y_pred, task, method, confidence_level, compute_ci, plot, **kwargs
            Same as evaluate() method
        metrics : List[str]
            List of metric names to evaluate
            
        Returns:
        --------
        Dict[str, Union[float, Tuple[float, Tuple[float, float]]]]
            Dictionary with metric names as keys and results as values
            
        Example:
        --------
        >>> evaluator = MetricEvaluator()
        >>> results = evaluator.evaluate_multiple(
        ...     y_true=[1, 2, 3, 4, 5],
        ...     y_pred=[1.1, 2.1, 2.9, 4.1, 4.9],
        ...     task='regression',
        ...     metrics=['mae', 'rmse', 'r2'],
        ...     method='jackknife'
        ... )
        >>> for metric, (value, ci) in results.items():
        ...     print(f"{metric}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        """
        results = {}
        for metric in metrics:
            results[metric] = self.evaluate(
                y_true=y_true,
                y_pred=y_pred,
                task=task,
                metric=metric,
                method=method,
                confidence_level=confidence_level,
                compute_ci=compute_ci,
                plot=plot,
                **kwargs
            )
        return results
    
    def info(self) -> Dict[str, Any]:
        """
        Get information about available tasks, metrics, and methods.
        
        Returns:
        --------
        Dict[str, Any]: Dictionary containing all available options
        """
        return {
            'tasks': ['classification', 'regression'],
            'classification': {
                'metrics': list(self.classification_metrics.keys()),
                'methods': self.classification_methods
            },
            'regression': {
                'metrics': list(self.regression_metrics.keys()), 
                'methods': self.regression_methods
            }
        }
    
    def print_info(self):
        """Print formatted information about available options."""
        print("MetricEvaluator - Available Options")
        print("=" * 40)
        
        print("\n📊 CLASSIFICATION METRICS:")
        for metric in self.classification_metrics.keys():
            print(f"   - {metric}")
            
        print(f"\n🔧 Classification Methods: {len(self.classification_methods)}")
        for method in self.classification_methods:
            print(f"   - {method}")
        
        print("\n📈 REGRESSION METRICS:")
        for metric in self.regression_metrics.keys():
            print(f"   - {metric}")
            
        print(f"\n🔧 Regression Methods: {len(self.regression_methods)}")  
        for method in self.regression_methods:
            print(f"   - {method}")
        
        print(f"\nDefault method: bootstrap_bca")
        print(f"Default confidence level: 0.95")


# Convenience function for direct evaluation
def evaluate_metric(y_true: List[Union[int, float]], 
                   y_pred: List[Union[int, float]], 
                   task: Union[str, TaskType],
                   metric: str,
                   method: str = 'bootstrap_bca',
                   confidence_level: float = 0.95,
                   compute_ci: bool = True,
                   plot: bool = False,
                   **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Convenience function to evaluate a single metric with confidence intervals.
    
    This is a shortcut for MetricEvaluator().evaluate() for quick one-off evaluations.
    
    Parameters and returns are the same as MetricEvaluator.evaluate()
    
    Example:
    --------
    >>> from confidenceinterval import evaluate_metric
    >>> 
    >>> # Quick regression evaluation  
    >>> mae_val, ci = evaluate_metric(
    ...     y_true=[1, 2, 3, 4, 5],
    ...     y_pred=[1.1, 2.1, 2.9, 4.1, 4.9],
    ...     task='regression', 
    ...     metric='mae'
    ... )
    """
    evaluator = MetricEvaluator()
    return evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred, 
        task=task,
        metric=metric,
        method=method,
        confidence_level=confidence_level,
        compute_ci=compute_ci,
        plot=plot,
        **kwargs
    )