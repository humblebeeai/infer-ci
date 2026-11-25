""" Confidence intervals for common binary metrics."""

from ast import Call
import statsmodels
from statsmodels.stats.proportion import proportion_confint
from typing import List, Callable, Tuple, Union, Optional
from functools import partial
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from .utils import get_positive_negative_counts
from .methods import bootstrap_ci, bootstrap_methods, BootstrapParams
from .visualize import bootstrap_with_plot

proportion_conf_methods: List[str] = [
    'wilson',
    'normal',
    'agresti_coull',
    'beta',
    'jeffreys',
    'binom_test']





def accuracy_score_binomial_ci(y_true: List[int],
                               y_pred: List[int],
                               confidence_level: float = 0.95,
                               method: str = 'wilson',
                               compute_ci: bool = True
                               ) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the accuracy score and the confidence interval.
        The confidence interval is computed as a binomial proportion,
        for more information see
        https://www.statsmodels.org/devel/generated/statsmodels.stats.proportion.proportion_confint.html

    Parameters
    ----------
    y_true : List[int]
        The grount truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The method for the stats model proportion method, by default 'wilson'
    compute_ci : bool, optional
        If true return the confidence interval as well as the accuract score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The accuracy score and optionally the confidence interval.
    """

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'

    correct = np.sum(np.array(y_pred) == np.array(y_true))
    acc = correct / len(y_pred)
    if compute_ci:
        interval = proportion_confint(
            correct,
            len(y_pred),
            alpha=1 - confidence_level,
            method=method)
        return acc, interval
    else:
        return acc


def accuracy_score_bootstrap(y_true: List[int],
                             y_pred: List[int],
                             confidence_level: float = 0.95,
                             method: str = 'bootstrap_bca',
                             n_resamples: int = 9999,
                             random_state: Optional[np.random.RandomState] = None,
                             plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    """Compute the accuray score confidence interval using the bootstrap method.

    Parameters
    ----------
    y_true : List[int]
        The grount truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence interval level , by default 0.95
    method : str, optional
        The bootstrapping method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducability, by default None
    plot : bool, optional
        Whether to create histogram plot, by default False

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The accuracy score and the confidence interval.
    """

    accuracy_score_no_ci = partial(
        accuracy_score_binomial_ci,
        compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=accuracy_score_no_ci,
        metric_name="accuracy",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def accuracy_score(y_true: List[int],
                   y_pred: List[int],
                   confidence_level: float = 0.95,
                   method: str = 'wilson',
                   compute_ci: bool = True,
                   **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
        Compute the accuracy score and optionally the confidence interval.
        Parameters
        ----------
        y_true : List[int]
            The grount truth labels.
        y_pred : List[int]
            The predicted categories.
        confidence_level : float, optional
            The confidence interval level, by default 0.95
        method : str, optional
            The method for the stats model proportion method, by default 'wilson'
        compute_ci : bool, optional
            If true return the confidence interval as well as the accuract score, by default True

        Returns
        -------
        Union[float, Tuple[float, Tuple[float, float]]]
            The accuracy score and optionally the confidence interval.
    """
    if method in bootstrap_methods:
        return accuracy_score_bootstrap(
            y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return accuracy_score_binomial_ci(
            y_true, y_pred, confidence_level, method, compute_ci)


def ppv_score_binomial_ci(y_true: List[int],
                          y_pred: List[int],
                          confidence_level: float = 0.95,
                          method: str = 'wilson',
                          compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FP = TP[1], FP[1]

    result = TP / (TP + FP + 1e-7)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def ppv_score_bootstrap(y_true: List[int],
                        y_pred: List[int],
                        confidence_level: float = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples: int = 9999,
                        random_state: Optional[np.random.RandomState] = None,
                        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    ppv_score_no_ci = partial(ppv_score_binomial_ci, compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=ppv_score_no_ci,
        metric_name="precision",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def ppv_score(y_true: List[int],
              y_pred: List[int],
              confidence_level: float = 0.95,
              method: str = 'wilson',
              compute_ci: bool = True,
              **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return ppv_score_bootstrap(
            y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return ppv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, compute_ci)


def npv_score_binomial_ci(y_true: List[int],
                          y_pred: List[int],
                          confidence_level: float = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, Tuple[float, float]]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FN = TN[1], FN[1]

    result = TN / (TN + FN + 1e-7)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def npv_score_bootstrap(y_true: List[int],
                        y_pred: List[int],
                        confidence_level: float = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples: int = 9999,
                        random_state: Optional[np.random.RandomState] = None,
                        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    npv_score_no_ci = partial(npv_score_binomial_ci, compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=npv_score_no_ci,
        metric_name="npv",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def npv_score(y_true: List[int],
              y_pred: List[int],
              confidence_level: float = 0.95,
              method: str = 'wilson',
              compute_ci: bool = True,
              **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return npv_score_bootstrap(
            y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return npv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, compute_ci)


def tpr_score_binomial_ci(y_true: List[int],
                          y_pred: List[int],
                          confidence_level: float = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, Tuple[float, float]]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FN = TP[1], FN[1]

    result = TP / (TP + FN + 1e-7)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tpr_score_bootstrap(y_true: List[int],
                        y_pred: List[int],
                        confidence_level: float = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples: int = 9999,
                        random_state: Optional[np.random.RandomState] = None,
                        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    tpr_score_no_ci = partial(tpr_score_binomial_ci, compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=tpr_score_no_ci,
        metric_name="recall",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def tpr_score(y_true: List[int],
              y_pred: List[int],
              confidence_level: float = 0.95,
              method: str = 'wilson',
              compute_ci: bool = True,
              **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return tpr_score_bootstrap(
            y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return tpr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, compute_ci=compute_ci)


def fpr_score_binomial_ci(y_true: List[int],
                          y_pred: List[int],
                          confidence_level: float = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, Tuple[float, float]]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    FP, TN = FP[1], TN[1]

    result = FP / (FP + TN + 1e-7)
    if compute_ci:
        interval = proportion_confint(
            FP, FP + TN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def fpr_score_bootstrap(y_true: List[int],
                        y_pred: List[int],
                        confidence_level: float = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples: int = 9999,
                        random_state: Optional[np.random.RandomState] = None,
                        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    fpr_score_no_ci = partial(fpr_score_binomial_ci, compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=fpr_score_no_ci,
        metric_name="fpr",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def fpr_score(y_true: List[int],
              y_pred: List[int],
              confidence_level: float = 0.95,
              method: str = 'wilson',
              compute_ci: bool = True,
              **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return fpr_score_bootstrap(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, **kwargs)
    else:
        return fpr_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, compute_ci=compute_ci)


def tnr_score_binomial_ci(y_true: List[int],
                          y_pred: List[int],
                          confidence_level: float = 0.95,
                          method: str = 'wilson',
                          compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FP = TN[1], FP[1]

    result = TN / (TN + FP + 1e-7)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tnr_score_bootstrap(y_true: List[int],
                        y_pred: List[int],
                        confidence_level: float = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples: int = 9999,
                        random_state: Optional[np.random.RandomState] = None,
                        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    tnr_score_no_ci = partial(tnr_score_binomial_ci, compute_ci=False)

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=tnr_score_no_ci,
        metric_name="specificity",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def tnr_score(y_true: List[int],
              y_pred: List[int],
              confidence_level: float = 0.95,
              method: str = 'wilson',
              compute_ci: bool = True,
              **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return tnr_score_bootstrap(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, **kwargs)
    else:
        return tnr_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, compute_ci=compute_ci)


def jaccard_score_binomial_ci(y_true: List[int],
                             y_pred: List[int],
                             confidence_level: float = 0.95,
                             method: str = 'wilson',
                             compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the Jaccard score (TP / (TP + FP + FN)) and optionally the confidence interval using binomial proportion methods."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays y_true and y_pred must not be empty.")

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if not set(np.unique(y_true_arr)).issubset({0, 1}) or not set(np.unique(y_pred_arr)).issubset({0, 1}):
        raise ValueError("Inputs y_true and y_pred must contain only binary values (0 or 1).")

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FP, FN = TP[1], FP[1], FN[1]

    trials = TP + FP + FN

    if trials == 0:
        raise ValueError("TP + FP + FN is 0. Cannot compute Jaccard score when there are no positives or positive predictions.")

    result = float(TP / trials)
    if compute_ci:
        interval = proportion_confint(
            TP, trials, alpha=1 - confidence_level, method=method)
        return result, (float(interval[0]), float(interval[1]))
    else:
        return result


def jaccard_score_bootstrap(y_true: List[int],
                            y_pred: List[int],
                            confidence_level: float = 0.95,
                            method: str = 'bootstrap_bca',
                            n_resamples: int = 9999,
                            random_state: Optional[np.random.RandomState] = None,
                            plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    """Compute the Jaccard score confidence interval using the bootstrap method."""
    def jaccard_no_ci(yt, yp):
        yt_arr = np.asarray(yt)
        yp_arr = np.asarray(yp)
        tp = np.sum((yt_arr == 1) & (yp_arr == 1))
        fp = np.sum((yt_arr == 0) & (yp_arr == 1))
        fn = np.sum((yt_arr == 1) & (yp_arr == 0))
        denom = tp + fp + fn
        return float(tp / denom) if denom > 0 else 0.0

    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=jaccard_no_ci,
        metric_name="jaccard",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot
    )


def jaccard_score(y_true: List[int],
                  y_pred: List[int],
                  confidence_level: float = 0.95,
                  method: str = 'wilson',
                  compute_ci: bool = True,
                  **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the Jaccard score and optionally the confidence interval."""
    if not compute_ci:
        return jaccard_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method='wilson', compute_ci=False)

    if method in bootstrap_methods:
        return jaccard_score_bootstrap(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, **kwargs)
    else:
        return jaccard_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, compute_ci=True)
