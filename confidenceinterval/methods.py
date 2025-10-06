from scipy.stats import bootstrap
from scipy import stats
import numpy as np
from typing import List, Callable, Optional, Tuple

bootstrap_methods = [
    'bootstrap_bca',
    'bootstrap_percentile',
    'bootstrap_basic']

# Available methods: ['bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic', 'jackknife']
regression_conf_methods: List[str] = bootstrap_methods + ['jackknife']


class BootstrapParams:
    n_resamples: int
    random_state: Optional[np.random.RandomState]


def bootstrap_ci(y_true: List[int],
                 y_pred: List[int],
                 metric: Callable,
                 confidence_level: float = 0.95,
                 n_resamples: int = 9999,
                 method: str = 'bootstrap_bca',
                 random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        return metric(np.array(y_true)[indices], np.array(y_pred)[indices])

    assert method in bootstrap_methods, f'Bootstrap ci method {method} not in {bootstrap_methods}'

    indices = (np.arange(len(y_true)), )
    bootstrap_res = bootstrap(indices,
                              statistic=statistic,
                              n_resamples=n_resamples,
                              confidence_level=confidence_level,
                              method=method.split('bootstrap_')[1],
                              random_state=random_state)
    result = metric(y_true, y_pred)
    ci = bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high
    return result, ci


def jackknife_ci(y_true: List[float], 
                 y_pred: List[float], 
                 metric: Callable,
                 confidence_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Compute jackknife confidence interval for a regression metric.
    
    Parameters:
    -----------
    y_true : List[float]
        The ground truth target values
    y_pred : List[float] 
        The predicted target values
    metric : Callable
        Function that computes the metric from y_true and y_pred
    confidence_level : float
        Confidence level (default 0.95)
    
    Returns:
    --------
    Tuple[float, Tuple[float, float]]: (metric_value, (lower_bound, upper_bound))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    # Compute the original metric
    original_metric = metric(y_true, y_pred)
    
    # Compute jackknife samples (leave-one-out)
    jackknife_metrics = []
    for i in range(n):
        # Remove the i-th element
        y_true_jack = np.delete(y_true, i)
        y_pred_jack = np.delete(y_pred, i)
        jack_metric = metric(y_true_jack, y_pred_jack)
        jackknife_metrics.append(jack_metric)
    
    jackknife_metrics = np.array(jackknife_metrics)
    
    # Compute pseudo-values
    pseudo_values = n * original_metric - (n - 1) * jackknife_metrics
    
    # Compute standard error using pseudo-values
    pseudo_mean = np.mean(pseudo_values)
    pseudo_var = np.var(pseudo_values, ddof=1)
    se = np.sqrt(pseudo_var / n)
    
    # Compute confidence interval using t-distribution
    alpha = 1 - confidence_level
    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    
    lower = pseudo_mean - t_val * se
    upper = pseudo_mean + t_val * se
    
    return original_metric, (lower, upper)
