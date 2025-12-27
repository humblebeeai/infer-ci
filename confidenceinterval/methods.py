from scipy.stats import bootstrap
from scipy import stats
import numpy as np
from typing import List, Callable, Optional, Tuple, Union
from tqdm import tqdm

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
                 random_state: Optional[np.random.RandomState] = None,
                 return_samples: bool = False,
                 plot: bool = False,
                 show_progress: bool = True) -> Union[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float], np.ndarray]]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : List[int]
        The ground truth labels
    y_pred : List[int] 
        The predicted labels
    metric : Callable
        Function that computes the metric from y_true and y_pred
    confidence_level : float, optional
        Confidence level, by default 0.95
    n_resamples : int, optional
        Number of bootstrap resamples, by default 9999
    method : str, optional
        Bootstrap method, by default 'bootstrap_bca'
    random_state : Optional[np.random.RandomState], optional
        Random state for reproducibility, by default None
    return_samples : bool, optional
        Whether to return bootstrap samples, by default False
    plot : bool, optional
        Whether plotting is intended (automatically sets return_samples=True), by default False
    show_progress : bool, optional
        Whether to show progress bar, by default True

    Returns
    -------
    Union[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float], np.ndarray]]
        (metric_value, (lower, upper)) or (metric_value, (lower, upper), bootstrap_samples)
    """

    # Progress bar setup
    # For BCA method, total iterations = n_resamples + len(data) + 1
    # (bootstrap samples + jackknife for acceleration + original)
    # For other methods, total iterations = n_resamples
    if method == 'bootstrap_bca':
        # BCA requires additional jackknife iterations for acceleration constant
        # Total = n_resamples + n_data_points + 1
        n_data_points = len(y_true)
        total_iterations = n_resamples + n_data_points + 1
        bar_format = '{desc} (BCA): {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    else:
        total_iterations = n_resamples
        bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

    pbar = tqdm(total=total_iterations, desc="Bootstrap CI", disable=not show_progress,
                bar_format=bar_format)

    call_count = [0]  # Use list to allow modification in nested function

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        result = metric(np.array(y_true)[indices], np.array(y_pred)[indices])

        # Update progress bar
        call_count[0] += 1
        pbar.update(1)

        return result

    assert method in bootstrap_methods, f'Bootstrap ci method {method} not in {bootstrap_methods}'

    # If plot=True, automatically enable return_samples
    if plot:
        return_samples = True

    indices = (np.arange(len(y_true)), )
    bootstrap_res = bootstrap(indices,
                              statistic=statistic,
                              n_resamples=n_resamples,
                              confidence_level=confidence_level,
                              method=method.split('bootstrap_')[1],
                              random_state=random_state)

    pbar.close()

    result = metric(y_true, y_pred)
    ci = bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high

    if return_samples:
        # Get bootstrap samples for plotting
        bootstrap_samples = bootstrap_res.bootstrap_distribution
        return result, ci, bootstrap_samples
    else:
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
