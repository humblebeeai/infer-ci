"""Confidence intervals for regression metrics"""

from typing import List, Callable, Tuple, Union, Optional
from ast import Call
import numpy as np
from functools import partial
from .methods import bootstrap_ci, bootstrap_methods, jackknife_ci, regression_conf_methods



# Defining the confidence interval computation function

def _compute_confidence_interval(y_true: List[float],
                                 y_pred: List[float], 
                                 metric_func: Callable,
                                 confidence_level: float = 0.95,
                                 method: str = 'bootstrap_bca',
                                 **kwargs) -> Tuple[float, Tuple[float, float]]:
    """
    Generic helper function to compute confidence intervals for any metric.
    
    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    metric_func : Callable
        Function that computes the metric from y_true and y_pred.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The method to use ('jackknife' or bootstrap methods), by default 'bootstrap_bca'
    **kwargs
        Additional arguments passed to bootstrap_ci
        
    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The metric value and confidence interval (lower, upper)
    """
    if method == 'jackknife':
        return jackknife_ci(y_true, y_pred, metric_func, confidence_level)
    elif method in bootstrap_methods:
        return bootstrap_ci(y_true=y_true,
                            y_pred=y_pred,
                            metric=metric_func,
                            confidence_level=confidence_level,
                            n_resamples=kwargs.get('n_resamples', 9999),
                            method=method,
                            random_state=kwargs.get('random_state', None))
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: {regression_conf_methods}")


def _compute_metric_with_optional_ci(y_true: List[float],
                                     y_pred: List[float],
                                     metric_func: Callable,
                                     confidence_level: float = 0.95,
                                     method: str = 'bootstrap_bca',
                                     compute_ci: bool = True,
                                     **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Generic function to compute a metric with optional confidence interval.
    
    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    metric_func : Callable
        Function that computes the metric from y_true and y_pred.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The method to use, by default 'bootstrap_bca'
    compute_ci : bool, optional
        Whether to compute confidence interval, by default True
    **kwargs
        Additional arguments passed to bootstrap_ci
        
    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric value or (metric_value, (lower, upper)) if compute_ci=True
    """
    if compute_ci:
        return _compute_confidence_interval(y_true, y_pred, metric_func, confidence_level, method, **kwargs)
    else:
        return metric_func(y_true, y_pred)


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
    def mae_metric(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=mae_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )


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
    def mse_metric(y_true, y_pred):
        return np.mean(np.square(np.array(y_true) - np.array(y_pred)))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=mse_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )


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
    def rmse_metric(y_true, y_pred):
        return np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=rmse_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )

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
    def r2_metric(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        ss_res = np.sum(np.square(y_true_arr - y_pred_arr))
        ss_tot = np.sum(np.square(y_true_arr - np.mean(y_true_arr)))
        return 1 - (ss_res / (ss_tot + 1e-15))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=r2_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )


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
    def mape_metric(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        return np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-15))) * 100
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=mape_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )

def adjusted_r2_score(num_features: int,
                      y_true: List[float],
                      y_pred: List[float],
                      confidence_level: float = 0.95,
                      method: str = 'bootstrap_bca',
                      compute_ci: bool = True,
                      **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Adjusted Coefficient of Determination (R²) and optionally confidence interval.

    Parameters
    ----------
    num_features : int
        The number of features used in the model.
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values. 
    confidence_level: float, optional
        The confidence interval level, by default 0.95
    method: str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci: bool, optional
        If true return the confidence interval as well as Adjusted R² score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The Adjusted R² score and optionally the confidence interval.    
    """
    def adjusted_r2_metric(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        ss_nominator = np.sum(np.square(y_true_arr - y_pred_arr))
        ss_denominator = np.sum(np.square(y_true_arr - np.mean(y_true_arr)))
        r2_val = 1 - (ss_nominator/(ss_denominator + 1e-15))
        n = len(y_true)
        return 1 - (1 - r2_val) * (n - 1) / (n - num_features - 1 + 1e-15)
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=adjusted_r2_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )


def sym_mean_abs_per_error(
        y_true: List[float], 
        y_pred: List[float],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE) and optionally the confidence interval 

    Parameters: 
    ----------
    y_true: List[float]
        The ground truth target values.
    y_pred: List[float]
        The predicted target values. 
    confidence_level: float, optional
        The confidence interval level, by default 0.95
    method: str, optional 
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci: bool, optional 
        If true return the confidence interval as well as SMAPE score, by default True 

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The SMAPE score and optionally the confidence interval. 
    """
    def smape_metric(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        return np.mean(2 * np.abs(y_pred_arr - y_true_arr) / (np.abs(y_true_arr) + np.abs(y_pred_arr) + 1e-15)) * 100
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=smape_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )

def rmse_log(y_pred: List[float],
             y_true: List[float],
             confidence_level: float = 0.95,
             method: str = 'bootstrap_bca',
             compute_ci: bool = True,
             **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE) and optionally the confidence interval.

    Parameters
    ----------
    y_pred: List[float]
        The predicted target values.
    y_true: List[float]
        The ground truth target values.
    confidence_level: float, optional
        The confidence interval level, by default 0.95
    method: str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci: bool, optional
        If true return the confidence interval as well as RMSLE score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The RMSLE score and optionally the confidence interval.
    """
    def rmse_log_metric(y_true, y_pred):
        return np.sqrt(np.mean(np.square(np.log1p(np.array(y_true)) - np.log1p(np.array(y_pred)))))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=rmse_log_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )

def med_abs_err(y_true: List[float],
                y_pred: List[float],
                confidence_level: float = 0.95,
                method: str = 'bootstrap_bca',
                compute_ci: bool = True,
                **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Median Absolute Error and optionally the confidence interval.

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
        If true return the confidence interval as well as the Median Absolute Error score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The Median Absolute Error score and optionally the confidence interval.
    """
    def med_abs_err_metric(y_true, y_pred):
        return np.median(np.abs(np.array(y_true) - np.array(y_pred)))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=med_abs_err_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )

def huber_loss(y_true: List[float],
               y_pred: List[float],
               delta: float = 1.0,
               confidence_level: float = 0.95,
               method: str = 'bootstrap_bca',
               compute_ci: bool = True,
               **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Huber Loss and optionally the confidence interval.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    delta : float, optional
        The point where the loss function changes from quadratic to linear, by default 1.0
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        If true return the confidence interval as well as the Huber Loss score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The Huber Loss score and optionally the confidence interval.
    """
    def huber_loss_metric(y_true, y_pred):
        error = np.array(y_true) - np.array(y_pred)
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=huber_loss_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )


def exp_var_score(y_true: List[float],
                  y_pred: List[float],
                  confidence_level: float = 0.95,
                  method: str = 'bootstrap_bca',
                  compute_ci: bool = True,
                  **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Explained Variance Score and optionally the confidence interval.

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
        If true return the confidence interval as well as the Explained Variance Score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The Explained Variance Score and optionally the confidence interval.
    """
    def exp_var_metric(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        numerator = np.var(y_true_arr - y_pred_arr)
        denominator = np.var(y_true_arr) + 1e-15
        return 1 - (numerator / denominator)
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=exp_var_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )
                         


def mean_bia_dev(y_true: List[float],
                 y_pred: List[float],
                 confidence_level: float = 0.95,
                 method: str = 'bootstrap_bca',
                 compute_ci: bool = True,
                 **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute the Mean Bias Deviation and optionally the confidence interval.

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
        If true return the confidence interval as well as the Mean Bias Deviation score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The Mean Bias Deviation score and optionally the confidence interval.
    """
    def mean_bia_dev_metric(y_true, y_pred):
        return np.mean(np.array(y_pred) - np.array(y_true))
    
    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=mean_bia_dev_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    ) 

