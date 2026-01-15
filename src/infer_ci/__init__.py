from .__version__ import __version__

# Core unified interface for confidence interval metrics
from .evaluator import MetricEvaluator, evaluate_metric, TaskType

# Import utility modules
from .methods import *
from .utils import *

# Individual metric imports (for backward compatibility and advanced usage)
from .binary_metrics import accuracy_score, \
    ppv_score, \
    npv_score, \
    tpr_score, \
    fpr_score, \
    tnr_score
from .takahashi_methods import precision_score, \
    recall_score, \
    f1_score
from .auc import roc_auc_score
from .regression_metrics import mae, \
    mse, \
    rmse,\
    r2_score,\
    mape, \
    adjusted_r2_score,\
    sym_mean_abs_per_error,\
    rmse_log,\
    med_abs_err,\
    huber_loss, \
    exp_var_score, \
    iou, \
    mean_bia_dev

# Detection metrics
from .object_detection_metrics import map, \
    map50, \
    precision, \
    recall

from .classification_report import classification_report_with_ci

# Export the main interface
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
    'exp_var_score', 'mean_bia_dev', 'iou',

    # Detection metrics
    'map', 'map50', 'precision', 'recall',

    # Utility modules
    'methods', 'utils'
]
