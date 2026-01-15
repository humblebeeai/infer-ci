"""Visualization utilities for confidence interval evaluation."""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List, Optional, Callable
from .methods import bootstrap_ci


def create_bootstrap_histogram_plot(bootstrap_samples: np.ndarray,
                                  metric_value: float,
                                  confidence_interval: Tuple[float, float],
                                  metric_name: str,
                                  method: str,
                                  confidence_level: float,
                                  plot_type: str = "classification") -> str:
    """
    Create a histogram plot of bootstrap samples for metrics and save it to the Results folder.

    Parameters
    ----------
    bootstrap_samples : np.ndarray
        Bootstrap samples for plotting
    metric_value : float
        The computed metric value
    confidence_interval : Tuple[float, float]
        The confidence interval bounds
    metric_name : str
        Name of the metric
    method : str
        Bootstrap method used
    confidence_level : float
        Confidence level
    plot_type : str, optional
        Type of plot ("classification" or "regression"), by default "classification"

    Returns
    -------
    str
        Path to the saved plot
    """
    # Create Results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    use_frequency = plot_type.lower() == "detection"

    # Set color scheme and parameters based on plot type
    if use_frequency:
        color = 'skyblue'
        bins = 30
        alpha = 0.75
        density = False
        ylabel = 'Frequency'
        title_suffix = " (Detection)"
    elif plot_type.lower() == "regression":
        color = 'skyblue'
        bins = 50
        alpha = 0.7
        density = True
        ylabel = 'Density'
        title_suffix = ""  # Regression plots don't specify type in title
    else:
        color = 'lightcoral'
        bins = 50
        alpha = 0.7
        density = True
        ylabel = 'Density'
        title_suffix = f" ({plot_type.title()})"

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_samples, bins=bins, alpha=alpha, density=density, color=color,
             edgecolor='black', label='Bootstrap Distribution')

    # Add vertical lines for metric value and confidence interval
    if use_frequency:
        plt.axvline(metric_value, color='red', linestyle='--', linewidth=2,
                    label=f'Mean {metric_name}')
        plt.axvline(confidence_interval[0], color='blue', linestyle='--', linewidth=2,
                    label='95% CI Lower Bound')
        plt.axvline(confidence_interval[1], color='blue', linestyle='--', linewidth=2,
                    label='95% CI Upper Bound')
    else:
        plt.axvline(metric_value, color='red', linestyle='--', linewidth=2,
                    label=f'{metric_name.upper()}: {metric_value:.4f}')
        plt.axvline(confidence_interval[0], color='orange', linestyle=':', linewidth=2,
                    label=f'{confidence_level*100:.0f}% CI Lower: {confidence_interval[0]:.4f}')
        plt.axvline(confidence_interval[1], color='orange', linestyle=':', linewidth=2,
                    label=f'{confidence_level*100:.0f}% CI Upper: {confidence_interval[1]:.4f}')

    if not use_frequency:
        plt.axvspan(confidence_interval[0], confidence_interval[1], alpha=0.2, color='orange')

    # Formatting
    plt.xlabel(f'{metric_name}')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} Bootstrap Distribution{title_suffix}\n'
              f'Method: {method}, 95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{metric_name}_{method}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def bootstrap_with_plot(y_true: List[Union[int, float]],
                       y_pred: List[Union[int, float]],
                       metric_func: Callable,
                       metric_name: str,
                       confidence_level: float = 0.95,
                       method: str = 'bootstrap_bca',
                       n_resamples: int = 9999,
                       random_state: Optional[np.random.RandomState] = None,
                       plot: bool = False,
                       plot_type: str = "auto") -> Tuple[float, Tuple[float, float]]:
    """
    Generic function to compute bootstrap confidence intervals with optional plotting.

    Parameters
    ----------
    y_true : List[Union[int, float]]
        The ground truth labels/values
    y_pred : List[Union[int, float]]
        The predicted labels/values
    metric_func : Callable
        The metric function to bootstrap (should not compute CI itself)
    metric_name : str
        Name of the metric for plotting
    confidence_level : float, optional
        The confidence level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducibility, by default None
    plot : bool, optional
        Whether to create histogram plot, by default False
    plot_type : str, optional
        Type of plot ("classification", "regression", or "auto"), by default "auto"
        If "auto", will determine based on data type: integers -> classification, floats -> regression

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The metric value and the confidence interval
    """
    # Auto-determine plot type if not specified
    if plot_type == "auto":
        # Check if y_true contains only integers
        if all(isinstance(val, (int, np.integer)) for val in y_true):
            plot_type = "classification"
        else:
            plot_type = "regression"

    if plot:
        # Get bootstrap samples for plotting
        result_with_samples = bootstrap_ci(y_true=y_true,
                                         y_pred=y_pred,
                                         metric=metric_func,
                                         confidence_level=confidence_level,
                                         n_resamples=n_resamples,
                                         method=method,
                                         random_state=random_state,
                                         return_samples=True)
        metric_value, ci, bootstrap_samples = result_with_samples

        # Create and save the histogram plot
        plot_path = create_bootstrap_histogram_plot(
            bootstrap_samples, metric_value, ci, metric_name, method, confidence_level, plot_type
        )
        print(f"Histogram plot saved to: {plot_path}")

        return metric_value, ci, plot_path
    else:
        return bootstrap_ci(y_true=y_true,
                           y_pred=y_pred,
                           metric=metric_func,
                           confidence_level=confidence_level,
                           n_resamples=n_resamples,
                           method=method,
                           random_state=random_state)
