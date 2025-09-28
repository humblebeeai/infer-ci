"""Confidence intervals for regression metrics"""

from typing import List, Callable, Tuple, Union, Optional
from ast import Call
import numpy as np
from functools import partial
from confidenceinterval.bootstrap import bootstrap_ci, bootstrap_methods
from scipy import stats


# Available methods: ['bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic', 'jackknife']
regression_conf_methods: List[str] = bootstrap_methods + ['jackknife']


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


def mae_bootstrap(y_true: List[float],
                  y_pred: List[float],
                  confidence_level: float = 0.95,
                  method: str = 'bootstrap_bca',
                  n_resamples: int = 9999,
                  random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:
    """Compute the Mean Absolute Error confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The MAE score and the confidence interval.
    """
    def mae_no_ci(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=mae_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def mae(y_true: List[float],
        y_pred: List[float],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        compute_ci: bool = True,
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

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The MAE score and optionally the confidence interval.
    """
    mae_score = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    if compute_ci:
        if method == 'jackknife':
            def mae_metric(y_true, y_pred):
                return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
            return jackknife_ci(y_true, y_pred, mae_metric, confidence_level)
        elif method in bootstrap_methods:
            return mae_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    return mae_score


def mse_bootstrap(y_true: List[float],
                  y_pred: List[float],
                  confidence_level: float = 0.95,
                  method: str = 'bootstrap_bca',
                  n_resamples: int = 9999,
                  random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:
    """Compute the Mean Squared Error confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The MSE score and the confidence interval.
    """
    def mse_no_ci(y_true, y_pred):
        return np.mean(np.square(np.array(y_true) - np.array(y_pred)))
    
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=mse_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def mse(y_true: List[float],
        y_pred: List[float],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Mean Squared Error and optionally the confidence interval.
    
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
        If true return the confidence interval as well as the MSE score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The MSE score and optionally the confidence interval.
    """
    mse_score = np.mean(np.square(np.array(y_true) - np.array(y_pred)))
    
    if compute_ci:
        if method == 'jackknife':
            def mse_metric(y_true, y_pred):
                return np.mean(np.square(np.array(y_true) - np.array(y_pred)))
            return jackknife_ci(y_true, y_pred, mse_metric, confidence_level)
        elif method in bootstrap_methods:
            return mse_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    return mse_score


def rmse_bootstrap(y_true: List[float],
                   y_pred: List[float],
                   confidence_level: float = 0.95,
                   method: str = 'bootstrap_bca',
                   n_resamples: int = 9999,
                   random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:
    """Compute the Root Mean Squared Error confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The RMSE score and the confidence interval.
    """
    def rmse_no_ci(y_true, y_pred):
        return np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))
    
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=rmse_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def rmse(y_true: List[float],
         y_pred: List[float],
         confidence_level: float = 0.95,
         method: str = 'bootstrap_bca',
         compute_ci: bool = True,
         **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Root Mean Squared Error and optionally the confidence interval.
    
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
        If true return the confidence interval as well as the RMSE score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The RMSE score and optionally the confidence interval.
    """
    rmse_score = np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))
    
    if compute_ci:
        if method == 'jackknife':
            def rmse_metric(y_true, y_pred):
                return np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))
            return jackknife_ci(y_true, y_pred, rmse_metric, confidence_level)
        elif method in bootstrap_methods:
            return rmse_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    return rmse_score


def r2_bootstrap(y_true: List[float],
                 y_pred: List[float],
                 confidence_level: float = 0.95,
                 method: str = 'bootstrap_bca',
                 n_resamples: int = 9999,
                 random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:
    """Compute the Coefficient of Determination (R²) confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The R² score and the confidence interval.
    """
    def r2_no_ci(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        ss_res = np.sum(np.square(y_true_arr - y_pred_arr))
        ss_tot = np.sum(np.square(y_true_arr - np.mean(y_true_arr)))
        return 1 - (ss_res / (ss_tot + 1e-15))  # Add small epsilon to avoid division by zero
    
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=r2_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def r2_score(y_true: List[float],
             y_pred: List[float],
             confidence_level: float = 0.95,
             method: str = 'bootstrap_bca',
             compute_ci: bool = True,
             **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Coefficient of Determination (R²) and optionally the confidence interval.
    
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
        If true return the confidence interval as well as the R² score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The R² score and optionally the confidence interval.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    ss_res = np.sum(np.square(y_true_arr - y_pred_arr))
    ss_tot = np.sum(np.square(y_true_arr - np.mean(y_true_arr)))
    r2_val = 1 - (ss_res / (ss_tot + 1e-15))  # Add small epsilon to avoid division by zero
    
    if compute_ci:
        if method == 'jackknife':
            def r2_metric(y_true, y_pred):
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                ss_res = np.sum(np.square(y_true_arr - y_pred_arr))
                ss_tot = np.sum(np.square(y_true_arr - np.mean(y_true_arr)))
                return 1 - (ss_res / (ss_tot + 1e-15))
            return jackknife_ci(y_true, y_pred, r2_metric, confidence_level)
        elif method in bootstrap_methods:
            return r2_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    return r2_val


def mape_bootstrap(y_true: List[float],
                   y_pred: List[float],
                   confidence_level: float = 0.95,
                   method: str = 'bootstrap_bca',
                   n_resamples: int = 9999,
                   random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:
    """Compute the Mean Absolute Percentage Error confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The MAPE score and the confidence interval.
    """
    def mape_no_ci(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        # Add small epsilon to avoid division by zero
        return np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-15))) * 100
    
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=mape_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def mape(y_true: List[float],
         y_pred: List[float],
         confidence_level: float = 0.95,
         method: str = 'bootstrap_bca',
         compute_ci: bool = True,
         **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Mean Absolute Percentage Error and optionally the confidence interval.
    
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
        If true return the confidence interval as well as the MAPE score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The MAPE score and optionally the confidence interval.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    # Add small epsilon to avoid division by zero
    mape_score = np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-15))) * 100
    
    if compute_ci:
        if method == 'jackknife':
            def mape_metric(y_true, y_pred):
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                return np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-15))) * 100
            return jackknife_ci(y_true, y_pred, mape_metric, confidence_level)
        elif method in bootstrap_methods:
            return mape_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    return mape_score

