"""Confidence intervals for object detection metrics"""

from typing import Dict, Union, Tuple, Optional, Any, List, Callable
import numpy as np
import sys
from pathlib import Path

# Add the customized ultralytics to path
ultralytics_path = Path(__file__).parent.parent / "ultralytics"
if str(ultralytics_path.parent) not in sys.path:
    sys.path.insert(0, str(ultralytics_path.parent))

try:
    from ultralytics.utils.metrics import ap_per_class, compute_ap
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    ap_per_class = None
    compute_ap = None

from .methods import bootstrap_ci, bootstrap_methods, regression_conf_methods


class DetectionData:
    """
    Container for detection evaluation data.

    This stores the raw detection statistics needed to compute metrics:
    - tp: True positive array (shape: [n_detections, n_iou_thresholds])
    - conf: Confidence scores (shape: [n_detections])
    - pred_cls: Predicted classes (shape: [n_detections])
    - target_cls: Target (ground truth) classes (shape: [n_targets])
    - class_names: Dictionary mapping class indices to names
    """
    def __init__(self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                 target_cls: np.ndarray, class_names: Dict[int, str] = None):
        self.tp = tp
        self.conf = conf
        self.pred_cls = pred_cls
        self.target_cls = target_cls
        self.class_names = class_names or {}
        self.nc = len(np.unique(target_cls)) if len(target_cls) > 0 else 0

    def __len__(self):
        """Return number of detections."""
        return len(self.conf)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'tp': self.tp,
            'conf': self.conf,
            'pred_cls': self.pred_cls,
            'target_cls': self.target_cls,
            'class_names': self.class_names
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            tp=data['tp'],
            conf=data['conf'],
            pred_cls=data['pred_cls'],
            target_cls=data['target_cls'],
            class_names=data.get('class_names', {})
        )


def extract_detection_data(model: Any, data: str, **kwargs) -> DetectionData:
    """
    Run YOLO inference ONCE and extract detection statistics.

    This function performs a single inference run on the validation dataset
    and extracts all necessary statistics (tp, conf, pred_cls, target_cls)
    needed for computing detection metrics with confidence intervals.

    Parameters
    ----------
    model : Any
        YOLO model object or path to model weights (e.g., 'yolov8n.pt')
    data : str
        Path to dataset YAML configuration file (e.g., 'coco128.yaml')
    **kwargs
        Additional validation arguments (batch, imgsz, conf, iou, device, workers, etc.)

    Returns
    -------
    DetectionData
        Container with detection statistics:
        - tp: True positive array (shape: [n_detections, 10])  # 10 IoU thresholds 0.5:0.95
        - conf: Confidence scores (shape: [n_detections])
        - pred_cls: Predicted class indices (shape: [n_detections])
        - target_cls: Ground truth class indices (shape: [n_targets])
        - class_names: Dictionary mapping class index to name

    Example
    -------
    >>> from confidenceinterval.detection_metrics import extract_detection_data
    >>> from ultralytics import YOLO
    >>>
    >>> # Extract detection data (runs inference ONCE)
    >>> model = YOLO('yolov8n.pt')
    >>> det_data = extract_detection_data(model, 'coco128.yaml', batch=32)
    >>>
    >>> # Now use for metrics with bootstrap CI (no more inference needed!)
    >>> map_val, ci = yolo_map_with_ci(det_data, method='bootstrap_bca')

    Notes
    -----
    - Inference runs ONLY ONCE (efficient!)
    - Works with any ultralytics YOLO version
    - Returns data in format compatible with bootstrap resampling
    - No custom validator needed - uses standard ultralytics validation
    """

    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Ultralytics YOLO is not available. "
            f"Please ensure the 'ultralytics' directory exists at: {ultralytics_path}"
        )

    try:
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
    except ImportError as e:
        raise ImportError(
            f"Failed to import YOLO components: {e}. "
            "Please ensure Ultralytics is properly installed."
        )

    # Load model if path is provided
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    # Prepare validation arguments
    val_args = get_cfg(DEFAULT_CFG)
    val_args.mode = 'val'

    # Set data if provided
    if data:
        val_args.data = data

    # Update with all kwargs
    for key, value in kwargs.items():
        if hasattr(val_args, key):
            setattr(val_args, key, value)

    # Create validator and run validation ONCE
    validator = DetectionValidator(args=val_args)
    validator.data = validator.check_dataset(val_args.data)

    # Run validation
    validator(model=model.model)

    # Extract statistics from validator
    stats = {k: np.concatenate(v, 0) for k, v in validator.stats.items()}

    # Get class names
    class_names = validator.names

    # Create DetectionData container
    detection_data = DetectionData(
        tp=stats['tp'],
        conf=stats['conf'],
        pred_cls=stats['pred_cls'],
        target_cls=stats['target_cls'],
        class_names=class_names
    )

    return detection_data


def _compute_detection_metric_with_ci(
    detection_data: DetectionData,
    metric_func: Callable,
    confidence_level: float = 0.95,
    method: str = 'bootstrap_bca',
    compute_ci: bool = True,
    plot: bool = False,
    metric_name: str = "metric",
    **kwargs
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Generic function to compute detection metrics with optional confidence interval.

    This follows the same pattern as _compute_metric_with_optional_ci in regression_metrics.py
    but adapted for detection data.

    Parameters
    ----------
    detection_data : DetectionData
        Container with detection statistics (tp, conf, pred_cls, target_cls)
    metric_func : Callable
        Function that computes the metric from DetectionData
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The method to use, by default 'bootstrap_bca'
    compute_ci : bool, optional
        Whether to compute confidence interval, by default True
    plot : bool, optional
        Whether to create histogram plot (only for bootstrap methods), by default False
    metric_name : str, optional
        Name of the metric for plot labeling, by default "metric"
    **kwargs
        Additional arguments passed to bootstrap_ci (n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric value or (metric_value, (lower, upper)) if compute_ci=True
    """

    if not compute_ci:
        return metric_func(detection_data)

    # For bootstrap CI, we need to wrap the metric function to work with indices
    def statistic_wrapper(*indices):
        """Wrapper to resample detection data by indices."""
        indices = np.array(indices)[0, :]

        # Resample detection statistics
        resampled_data = DetectionData(
            tp=detection_data.tp[indices] if len(detection_data.tp) > 0 else detection_data.tp,
            conf=detection_data.conf[indices] if len(detection_data.conf) > 0 else detection_data.conf,
            pred_cls=detection_data.pred_cls[indices] if len(detection_data.pred_cls) > 0 else detection_data.pred_cls,
            target_cls=detection_data.target_cls,  # Ground truth doesn't get resampled
            class_names=detection_data.class_names
        )

        return metric_func(resampled_data)

    # Use scipy bootstrap
    from scipy.stats import bootstrap as scipy_bootstrap

    if method not in bootstrap_methods:
        raise ValueError(f"Unknown method: {method}. Available methods: {regression_conf_methods}")

    # Get number of detections for resampling
    n_detections = len(detection_data)
    if n_detections == 0:
        # No detections, return 0 with no CI
        return 0.0 if not compute_ci else (0.0, (0.0, 0.0))

    indices = (np.arange(n_detections), )
    bootstrap_res = scipy_bootstrap(
        indices,
        statistic=statistic_wrapper,
        n_resamples=kwargs.get('n_resamples', 9999),
        confidence_level=confidence_level,
        method=method.split('bootstrap_')[1],
        random_state=kwargs.get('random_state', None)
    )

    # Compute the actual metric value
    result = metric_func(detection_data)
    ci = bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high

    if plot:
        # TODO: Add plotting support similar to regression metrics
        print(f"Plotting for detection metrics not yet implemented")

    return result, ci


def yolo_map_with_ci(
    detection_data: DetectionData,
    confidence_level: float = 0.95,
    method: str = 'bootstrap_bca',
    compute_ci: bool = True,
    plot: bool = False,
    **kwargs
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mAP50-95 (mean Average Precision) with confidence intervals.

    This function follows the SAME pattern as regression metrics (mae, mse, etc.)
    but works on detection data extracted from YOLO model.

    Parameters
    ----------
    detection_data : DetectionData
        Detection statistics extracted from YOLO model (from extract_detection_data)
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
        Options: 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'
    compute_ci : bool, optional
        If true return the confidence interval as well as the mAP score, by default True
    plot : bool, optional
        If true create histogram plot for bootstrap methods, by default False
    **kwargs
        Additional arguments (n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP score and optionally the confidence interval (lower, upper)

    Example
    -------
    >>> from confidenceinterval import extract_detection_data, yolo_map_with_ci
    >>> from ultralytics import YOLO
    >>>
    >>> # Step 1: Run inference ONCE
    >>> model = YOLO('yolov8n.pt')
    >>> det_data = extract_detection_data(model, 'coco128.yaml')
    >>>
    >>> # Step 2: Compute mAP with CI (same pattern as regression!)
    >>> map_val, (ci_lower, ci_upper) = yolo_map_with_ci(
    ...     detection_data=det_data,
    ...     method='bootstrap_bca',
    ...     compute_ci=True
    ... )
    >>>
    >>> print(f"mAP: {map_val:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    Notes
    -----
    - No inference happens here! All inference done in extract_detection_data()
    - Bootstrap resampling happens at prediction level (fast!)
    - Same API as regression metrics (consistent pattern)
    - Works with any ultralytics version
    """

    def map_metric(det_data: DetectionData) -> float:
        """Compute mAP50-95 from detection data."""
        if len(det_data.conf) == 0:
            return 0.0

        # Use ultralytics ap_per_class function to compute metrics
        results = ap_per_class(
            tp=det_data.tp,
            conf=det_data.conf,
            pred_cls=det_data.pred_cls,
            target_cls=det_data.target_cls,
            plot=False,
            save_dir=Path(),
            names=det_data.class_names,
        )

        # results = (tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values)
        # ap has shape (nc, 10) - 10 IoU thresholds
        # mAP50-95 is the mean across all classes and all IoU thresholds
        ap = results[5]  # Average precision array (nc, 10)

        if len(ap) == 0:
            return 0.0

        # mAP50-95: mean across classes and IoU thresholds
        map_val = ap.mean()

        return float(map_val)

    return _compute_detection_metric_with_ci(
        detection_data=detection_data,
        metric_func=map_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="mAP50-95",
        **kwargs
    )


def yolo_map50_with_ci(
    detection_data: DetectionData,
    confidence_level: float = 0.95,
    method: str = 'bootstrap_bca',
    compute_ci: bool = True,
    plot: bool = False,
    **kwargs
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mAP50 (mean Average Precision at IoU=0.5) with confidence intervals.

    Parameters
    ----------
    detection_data : DetectionData
        Detection statistics from extract_detection_data
    confidence_level : float, optional
        Confidence interval level, by default 0.95
    method : str, optional
        Bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        Whether to compute CI, by default True
    plot : bool, optional
        Whether to create histogram plot, by default False
    **kwargs
        Additional arguments (n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP50 score and optionally the confidence interval
    """

    def map50_metric(det_data: DetectionData) -> float:
        """Compute mAP50 from detection data."""
        if len(det_data.conf) == 0:
            return 0.0

        results = ap_per_class(
            tp=det_data.tp,
            conf=det_data.conf,
            pred_cls=det_data.pred_cls,
            target_cls=det_data.target_cls,
            plot=False,
            save_dir=Path(),
            names=det_data.class_names,
        )

        ap = results[5]  # Average precision array (nc, 10)

        if len(ap) == 0:
            return 0.0

        # mAP50: mean across classes at IoU=0.5 (first column)
        map50_val = ap[:, 0].mean()

        return float(map50_val)

    return _compute_detection_metric_with_ci(
        detection_data=detection_data,
        metric_func=map50_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="mAP50",
        **kwargs
    )


def yolo_precision_with_ci(
    detection_data: DetectionData,
    confidence_level: float = 0.95,
    method: str = 'bootstrap_bca',
    compute_ci: bool = True,
    plot: bool = False,
    **kwargs
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mean precision with confidence intervals.

    Parameters
    ----------
    detection_data : DetectionData
        Detection statistics from extract_detection_data
    confidence_level : float, optional
        Confidence interval level, by default 0.95
    method : str, optional
        Bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        Whether to compute CI, by default True
    plot : bool, optional
        Whether to create histogram plot, by default False
    **kwargs
        Additional arguments (n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean precision and optionally the confidence interval
    """

    def precision_metric(det_data: DetectionData) -> float:
        """Compute mean precision from detection data."""
        if len(det_data.conf) == 0:
            return 0.0

        results = ap_per_class(
            tp=det_data.tp,
            conf=det_data.conf,
            pred_cls=det_data.pred_cls,
            target_cls=det_data.target_cls,
            plot=False,
            save_dir=Path(),
            names=det_data.class_names,
        )

        # results[2] is precision array (nc,)
        p = results[2]

        if len(p) == 0:
            return 0.0

        # Mean precision across all classes
        mean_precision = p.mean()

        return float(mean_precision)

    return _compute_detection_metric_with_ci(
        detection_data=detection_data,
        metric_func=precision_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="Precision",
        **kwargs
    )


def yolo_recall_with_ci(
    detection_data: DetectionData,
    confidence_level: float = 0.95,
    method: str = 'bootstrap_bca',
    compute_ci: bool = True,
    plot: bool = False,
    **kwargs
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mean recall with confidence intervals.

    Parameters
    ----------
    detection_data : DetectionData
        Detection statistics from extract_detection_data
    confidence_level : float, optional
        Confidence interval level, by default 0.95
    method : str, optional
        Bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        Whether to compute CI, by default True
    plot : bool, optional
        Whether to create histogram plot, by default False
    **kwargs
        Additional arguments (n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean recall and optionally the confidence interval
    """

    def recall_metric(det_data: DetectionData) -> float:
        """Compute mean recall from detection data."""
        if len(det_data.conf) == 0:
            return 0.0

        results = ap_per_class(
            tp=det_data.tp,
            conf=det_data.conf,
            pred_cls=det_data.pred_cls,
            target_cls=det_data.target_cls,
            plot=False,
            save_dir=Path(),
            names=det_data.class_names,
        )

        # results[3] is recall array (nc,)
        r = results[3]

        if len(r) == 0:
            return 0.0

        # Mean recall across all classes
        mean_recall = r.mean()

        return float(mean_recall)

    return _compute_detection_metric_with_ci(
        detection_data=detection_data,
        metric_func=recall_metric,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        plot=plot,
        metric_name="Recall",
        **kwargs
    )


# List of available detection metrics
detection_metrics = [
    'yolo_map_with_ci',
    'yolo_map50_with_ci',
    'yolo_precision_with_ci',
    'yolo_recall_with_ci',
]

# For consistency with other metric modules
detection_conf_methods = bootstrap_methods
