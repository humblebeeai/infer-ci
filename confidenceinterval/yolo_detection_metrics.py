"""Confidence intervals for YOLO object detection metrics

This module implements detection metrics (mAP, precision, recall) with confidence intervals
using bootstrap resampling at the image level. Compatible with any pip-installed ultralytics.

Usage:
    from ultralytics import YOLO
    from confidenceinterval import yolo_map

    # Run predictions once
    model = YOLO('yolov8n.pt')
    results = model.predict(source='val-dataset/images/')

    # Calculate mAP with CI (bootstrap resamples images, not detections)
    map_val, (lower, upper) = yolo_map(
        y_true='val-dataset',  # Path to validation dataset
        y_pred=results,         # Predictions from model.predict()
        method='bootstrap_bca',
        confidence_level=0.95
    )
"""

from typing import List, Tuple, Union, Dict, Any
import numpy as np
from pathlib import Path
import yaml

from .visualize import bootstrap_with_plot


# ============================================================================
# Helper Functions (Internal)
# ============================================================================

def _load_ground_truth(y_true: Union[str, Path]) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """
    Load ground truth boxes from validation dataset directory.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file

    Returns
    -------
    Tuple[List[np.ndarray], List[int], Dict[int, str]]
        - List of GT boxes per image, each as array [n_boxes, 5] with format [class, x_center, y_center, w, h]
        - List of image shapes (height, width) per image
        - Dictionary mapping class ID to class name
    """
    y_true = Path(y_true)

    # If y_true is a YAML file, load the path from it
    if y_true.suffix in ['.yaml', '.yml']:
        with open(y_true, 'r') as f:
            data = yaml.safe_load(f)
            dataset_path = Path(data.get('path', y_true.parent))
            names = data.get('names', {})
    else:
        dataset_path = y_true
        names = {}

    # Find labels directory
    labels_dir = dataset_path / 'labels'
    images_dir = dataset_path / 'images'

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load all label files
    label_files = sorted(labels_dir.glob('*.txt'))
    ground_truth_boxes = []
    image_shapes = []

    for label_file in label_files:
        # Read label file
        if label_file.stat().st_size == 0:
            # Empty file (no objects in image)
            ground_truth_boxes.append(np.zeros((0, 5)))
            # Try to get image shape
            image_file = images_dir / f"{label_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_dir / f"{label_file.stem}.png"
            if image_file.exists():
                from PIL import Image
                img = Image.open(image_file)
                image_shapes.append(img.size[::-1])  # (height, width)
            else:
                image_shapes.append((640, 640))  # Default
        else:
            boxes = np.loadtxt(label_file).reshape(-1, 5)
            ground_truth_boxes.append(boxes)

            # Get image shape from corresponding image
            image_file = images_dir / f"{label_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_dir / f"{label_file.stem}.png"
            if image_file.exists():
                from PIL import Image
                img = Image.open(image_file)
                image_shapes.append(img.size[::-1])  # (height, width)
            else:
                image_shapes.append((640, 640))  # Default

    return ground_truth_boxes, image_shapes, names


def _parse_predictions(y_pred: List[Any]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Parse predictions from ultralytics Results objects.

    Parameters
    ----------
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        - List of prediction arrays per image, each as [n_detections, 6]
          with format [x1, y1, x2, y2, confidence, class]
        - List of image filenames (stem only, without extension)
    """
    predictions = []
    filenames = []

    for result in y_pred:
        # Get filename from path
        image_path = Path(result.path)
        filenames.append(image_path.stem)

        if result.boxes is not None and len(result.boxes) > 0:
            # Extract boxes, confidence, and class
            boxes = result.boxes.xyxy.cpu().numpy()  # [n, 4] - x1y1x2y2 format
            conf = result.boxes.conf.cpu().numpy()    # [n]
            cls = result.boxes.cls.cpu().numpy()      # [n]

            # Concatenate to [n, 6]
            pred_array = np.column_stack([boxes, conf, cls])
            predictions.append(pred_array)
        else:
            # No detections in this image
            predictions.append(np.zeros((0, 6)))

    return predictions, filenames


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of boxes (optimized vectorized version).

    Parameters
    ----------
    box1 : np.ndarray
        Boxes in format [N, 4] as [x1, y1, x2, y2]
    box2 : np.ndarray
        Boxes in format [M, 4] as [x1, y1, x2, y2]

    Returns
    -------
    np.ndarray
        IoU matrix of shape [N, M]
    """
    # Pre-calculate areas to avoid redundant computation
    # Shape: [N]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    # Shape: [M]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Expand dimensions for broadcasting
    box1 = box1[:, None, :]  # [N, 1, 4]
    box2 = box2[None, :, :]  # [1, M, 4]

    # Calculate intersection (optimized with single clip operation)
    # [N, M, 2]
    inter_xy_min = np.maximum(box1[..., :2], box2[..., :2])
    inter_xy_max = np.minimum(box1[..., 2:], box2[..., 2:])
    inter_wh = np.clip(inter_xy_max - inter_xy_min, 0, None)

    # Calculate intersection area [N, M]
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Calculate union using pre-computed areas
    # Broadcasting: [N, 1] + [1, M] - [N, M] = [N, M]
    union_area = area1[:, None] + area2[None, :] - inter_area

    # Calculate IoU (avoid division by zero)
    iou = inter_area / np.maximum(union_area, 1e-10)

    return iou


def _convert_yolo_to_xyxy(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert YOLO format boxes to xyxy format.

    Parameters
    ----------
    boxes : np.ndarray
        Boxes in YOLO format [n, 5] as [class, x_center, y_center, width, height] (normalized 0-1)
    img_shape : Tuple[int, int]
        Image shape as (height, width)

    Returns
    -------
    np.ndarray
        Boxes in xyxy format [n, 5] as [x1, y1, x2, y2, class]
    """
    if len(boxes) == 0:
        return np.zeros((0, 5))

    h, w = img_shape
    boxes_xyxy = np.zeros((len(boxes), 5))

    # Extract YOLO format values
    cls = boxes[:, 0]
    x_center = boxes[:, 1] * w
    y_center = boxes[:, 2] * h
    width = boxes[:, 3] * w
    height = boxes[:, 4] * h

    # Convert to xyxy
    boxes_xyxy[:, 0] = x_center - width / 2   # x1
    boxes_xyxy[:, 1] = y_center - height / 2  # y1
    boxes_xyxy[:, 2] = x_center + width / 2   # x2
    boxes_xyxy[:, 3] = y_center + height / 2  # y2
    boxes_xyxy[:, 4] = cls

    return boxes_xyxy


def _process_batch(pred_boxes: np.ndarray,
                   gt_boxes: np.ndarray,
                   iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10)) -> Dict[str, np.ndarray]:
    """
    Process one image: match predictions to ground truth and compute TP/FP.

    Parameters
    ----------
    pred_boxes : np.ndarray
        Predicted boxes [n_pred, 6] as [x1, y1, x2, y2, conf, class]
    gt_boxes : np.ndarray
        Ground truth boxes [n_gt, 5] as [x1, y1, x2, y2, class]
    iou_thresholds : np.ndarray
        IoU thresholds to evaluate (default: 0.5:0.05:0.95)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'tp': True positives array [n_pred, n_iou_thresholds]
        - 'conf': Confidence scores [n_pred]
        - 'pred_cls': Predicted classes [n_pred]
        - 'gt_cls': Ground truth classes [n_gt]
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    n_iou = len(iou_thresholds)

    # Initialize results
    tp = np.zeros((n_pred, n_iou))

    if n_pred == 0:
        return {
            'tp': tp,
            'conf': np.array([]),
            'pred_cls': np.array([]),
            'gt_cls': gt_boxes[:, 4] if n_gt > 0 else np.array([])
        }

    # Extract info
    conf = pred_boxes[:, 4]
    pred_cls = pred_boxes[:, 5]
    gt_cls = gt_boxes[:, 4] if n_gt > 0 else np.array([])

    if n_gt == 0:
        # No ground truth - all predictions are false positives
        return {
            'tp': tp,
            'conf': conf,
            'pred_cls': pred_cls,
            'gt_cls': gt_cls
        }

    # Sort predictions by confidence (descending)
    sort_idx = np.argsort(-conf)
    pred_boxes = pred_boxes[sort_idx]
    conf = conf[sort_idx]
    pred_cls = pred_cls[sort_idx]

    # Calculate IoU between all predictions and ground truths
    iou = _box_iou(pred_boxes[:, :4], gt_boxes[:, :4])  # [n_pred, n_gt]

    # Track which GT boxes have been matched
    gt_matched = np.zeros((n_iou, n_gt), dtype=bool)

    # Match predictions to ground truth
    for i, pred in enumerate(pred_boxes):
        pred_class = pred[5]

        # Find GT boxes of the same class
        same_class = gt_cls == pred_class
        if not same_class.any():
            continue

        # Get IoU with same-class GT boxes
        pred_iou = iou[i] * same_class  # Zero out different classes

        # For each IoU threshold
        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            # Find best matching GT box that exceeds threshold
            valid_matches = pred_iou >= iou_thresh

            if valid_matches.any():
                # Get best match that hasn't been matched yet
                valid_matches = valid_matches & ~gt_matched[iou_idx]

                if valid_matches.any():
                    # Match to GT box with highest IoU
                    best_gt_idx = np.argmax(pred_iou * valid_matches)
                    tp[i, iou_idx] = 1
                    gt_matched[iou_idx, best_gt_idx] = True

    # Restore original order (sort back)
    reverse_idx = np.argsort(sort_idx)
    tp = tp[reverse_idx]

    return {
        'tp': tp,
        'conf': pred_boxes[:, 4],  # Already sorted
        'pred_cls': pred_boxes[:, 5],  # Already sorted
        'gt_cls': gt_cls
    }


def _smooth(y, f=0.05):
    """Box filter of fraction f (from ultralytics)."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def _compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def _calculate_ap(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                  gt_cls: np.ndarray, eps = 1e-16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Average Precision per class (matches ultralytics implementation).

    Parameters
    ----------
    tp : np.ndarray
        True positives array [n_detections, n_iou_thresholds]
    conf : np.ndarray
        Confidence scores [n_detections]
    pred_cls : np.ndarray
        Predicted classes [n_detections]
    gt_cls : np.ndarray
        Ground truth classes [n_gt_boxes]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - AP per class and IoU threshold [n_classes, n_iou_thresholds]
        - Precision per class [n_classes]
        - Recall per class [n_classes]
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(gt_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    # Process each class (like ultralytics lines 577-601)
    for ci, c in enumerate(unique_classes):
        # Get predictions for this class
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs (like ultralytics lines 585-586)
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall curve (like ultralytics line 589)
        recall = tpc / (n_l + eps)
        # Interpolate recall curve (negative x, xp because xp decreases)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision curve
        precision = tpc / (tpc + fpc)
        # Interpolate precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        # Calculate AP for each IoU threshold (like ultralytics lines 597-598)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = _compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 curves and find global max F1 point (like ultralytics lines 605, 614-615)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    # Find the GLOBAL max F1 index (same for all classes)
    i = _smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    # max-F1 precision, recall
    precision = p_curve[:, i]  # Shape: (nc,)
    recall = r_curve[:, i]     # Shape: (nc,)

    return ap, precision, recall


def _calculate_metrics_from_data(predictions: List[np.ndarray],
                                 ground_truths: List[np.ndarray],
                                 image_shapes: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Calculate all detection metrics from predictions and ground truth.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays per image [n_pred, 6] as [x1, y1, x2, y2, conf, class]
    ground_truths : List[np.ndarray]
        List of GT arrays per image [n_gt, 5] as [class, x_center, y_center, w, h] (YOLO format)
    image_shapes : List[Tuple[int, int]]
        List of image shapes (height, width)

    Returns
    -------
    Dict[str, float]
        Dictionary with metrics: 'map', 'map50', 'precision', 'recall'
    """
    # Convert GT from YOLO format to xyxy
    gt_xyxy = [_convert_yolo_to_xyxy(gt, shape) for gt, shape in zip(ground_truths, image_shapes)]

    # Collect all results
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_gt_cls = []

    for pred, gt in zip(predictions, gt_xyxy):
        batch_result = _process_batch(pred, gt)
        if len(batch_result['tp']) > 0:
            all_tp.append(batch_result['tp'])
            all_conf.append(batch_result['conf'])
            all_pred_cls.append(batch_result['pred_cls'])
        if len(batch_result['gt_cls']) > 0:
            all_gt_cls.append(batch_result['gt_cls'])

    # Concatenate all results
    if len(all_tp) > 0:
        tp = np.vstack(all_tp)
        conf = np.concatenate(all_conf)
        pred_cls = np.concatenate(all_pred_cls)
    else:
        tp = np.zeros((0, 10))
        conf = np.array([])
        pred_cls = np.array([])

    if len(all_gt_cls) > 0:
        gt_cls = np.concatenate(all_gt_cls)
    else:
        gt_cls = np.array([])

    # Calculate AP (n_classes is inferred from gt_cls inside the function)
    ap, precision, recall = _calculate_ap(tp, conf, pred_cls, gt_cls)

    # Calculate mAP@0.5:0.95 (mean over all classes and IoU thresholds)
    # Like ultralytics, we include ALL classes (even those with 0 AP)
    if len(ap) > 0:
        map_value = ap.mean()  # Mean over all classes and IoU thresholds
        map50_value = ap[:, 0].mean()  # First IoU threshold is 0.5
        precision_mean = precision.mean()
        recall_mean = recall.mean()
    else:
        map_value = 0.0
        map50_value = 0.0
        precision_mean = 0.0
        recall_mean = 0.0

    return {
        'map': map_value,
        'map50': map50_value,
        'precision': precision_mean,
        'recall': recall_mean
    }


def _prepare_detection_data(y_true: Union[str, Path],
                            y_pred: List[Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Prepare detection data by loading ground truth and matching to predictions.

    This helper eliminates code duplication across all YOLO metric functions.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory or YAML file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]
        - matched_predictions: List of prediction arrays per image
        - ground_truths: List of ground truth arrays per image
        - image_shapes: List of image shapes (height, width)
    """
    # Load all ground truth
    all_ground_truths, all_image_shapes, class_names = _load_ground_truth(y_true)

    # Parse predictions
    predictions, pred_filenames = _parse_predictions(y_pred)

    # Get dataset path and label filenames
    y_true_path = Path(y_true)
    if y_true_path.suffix in ['.yaml', '.yml']:
        with open(y_true_path, 'r') as f:
            data = yaml.safe_load(f)
            dataset_path = Path(data.get('path', y_true_path.parent))
    else:
        dataset_path = y_true_path

    labels_dir = dataset_path / 'labels'
    label_files = sorted(labels_dir.glob('*.txt'))
    label_filenames = [lf.stem for lf in label_files]

    # Match ground truth to predictions by filename
    matched_predictions = []
    ground_truths = []
    image_shapes = []
    skipped_files = []

    for i, pred_fname in enumerate(pred_filenames):
        if pred_fname in label_filenames:
            idx = label_filenames.index(pred_fname)
            matched_predictions.append(predictions[i])
            ground_truths.append(all_ground_truths[idx])
            image_shapes.append(all_image_shapes[idx])
        else:
            skipped_files.append(pred_fname)

    if len(skipped_files) > 0:
        print(f"  ⚠️  Skipped {len(skipped_files)} predictions without matching ground truth labels:")
        for fname in skipped_files:
            print(f"     - {fname}.txt (label file not found)")

    if len(matched_predictions) == 0:
        raise ValueError(
            "No matching ground truth found for any predictions. "
            "Please check that your ground truth labels match the predicted image filenames."
        )

    return matched_predictions, ground_truths, image_shapes


# ============================================================================
# Main Metric Functions (Public API)
# ============================================================================

def yolo_map(y_true: Union[str, Path],
             y_pred: List[Any],
             confidence_level: float = 0.95,
             method: str = 'bootstrap_percentile',
             compute_ci: bool = True,
             plot: bool = False,
             **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mAP@0.5:0.95 with confidence interval.

    This function calculates the mean Average Precision across IoU thresholds 0.5:0.05:0.95
    with confidence intervals using bootstrap resampling at the image level.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method ('bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'),
        by default 'bootstrap_percentile'
    compute_ci : bool, optional
        If True return the confidence interval as well as the metric score, by default True
    plot : bool, optional
        If True create histogram plot for bootstrap methods, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP@0.5:0.95 score and optionally the confidence interval

    Examples
    --------
    >>> from ultralytics import YOLO
    >>> from confidenceinterval import yolo_map
    >>>
    >>> model = YOLO('yolov8n.pt')
    >>> results = model.predict(source='val-dataset/images/')
    >>>
    >>> # Calculate mAP with CI
    >>> map_val, (lower, upper) = yolo_map(
    ...     y_true='val-dataset',
    ...     y_pred=results,
    ...     method='bootstrap_percentile',
    ...     n_resamples=1000
    ... )
    >>> print(f"mAP@0.5:0.95: {map_val:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - This is the standard approach for object detection metrics
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - Compatible with any pip-installed ultralytics version
    """
    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes = _prepare_detection_data(y_true, y_pred)

    # Define metric calculation function that resamples images
    def map_metric(image_indices_y_true, image_indices_y_pred):
        """Calculate mAP for a subset of images (for bootstrap).

        NOTE: For YOLO, we must resample IMAGES (not individual predictions) because:
        - Predictions are grouped by image
        - We need to match predictions to GTs on the SAME image
        - AP calculation requires image-level processing

        Parameters
        ----------
        image_indices_y_true : array-like of int
            Image indices to resample (0-based)
        image_indices_y_pred : array-like of int
            Same as image_indices_y_true (both are image indices)
        """
        # Use the first argument (both are the same - image indices)
        subset_preds = [predictions[i] for i in image_indices_y_true]
        subset_gts = [ground_truths[i] for i in image_indices_y_true]
        subset_shapes = [image_shapes[i] for i in image_indices_y_true]

        metrics = _calculate_metrics_from_data(subset_preds, subset_gts, subset_shapes)
        return metrics['map']

    # Calculate mAP with optional CI
    if not compute_ci:
        # No CI computation - force plot to False
        plot = False
        # Just calculate metric once on all data
        all_indices = list(range(len(predictions)))
        return map_metric(all_indices, all_indices)

    # Extract bootstrap parameters
    n_resamples = kwargs.get('n_resamples', 100)
    random_state = kwargs.get('random_state', 42)

    # Create image indices array for bootstrap resampling
    # bootstrap_ci will resample these indices, and map_metric will use them
    # to select which images' predictions/ground_truths to include
    image_indices = np.arange(len(predictions))

    # Use bootstrap_with_plot helper to reduce code duplication
    # Both y_true and y_pred are the same (image indices) because we resample images, not predictions
    return bootstrap_with_plot(
        y_true=image_indices,
        y_pred=image_indices,
        metric_func=map_metric,
        metric_name="mAP@0.5:0.95",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot,
        plot_type="detection"
    )


def yolo_map50(y_true: Union[str, Path],
               y_pred: List[Any],
               confidence_level: float = 0.95,
               method: str = 'bootstrap_percentile',
               compute_ci: bool = True,
               plot: bool = False,
               **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mAP@0.5 with confidence interval.

    This function calculates the mean Average Precision at IoU threshold 0.5
    with confidence intervals using bootstrap resampling at the image level.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method ('bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'),
        by default 'bootstrap_percentile'
    compute_ci : bool, optional
        If True return the confidence interval as well as the metric score, by default True
    plot : bool, optional
        If True create histogram plot for bootstrap methods, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP@0.5 score and optionally the confidence interval

    Examples
    --------
    >>> from ultralytics import YOLO
    >>> from confidenceinterval import yolo_map50
    >>>
    >>> model = YOLO('yolov8n.pt')
    >>> results = model.predict(source='val-dataset/images/')
    >>>
    >>> # Calculate mAP@0.5 with CI
    >>> map50_val, (lower, upper) = yolo_map50(
    ...     y_true='val-dataset',
    ...     y_pred=results,
    ...     method='bootstrap_percentile',
    ...     n_resamples=1000
    ... )
    >>> print(f"mAP@0.5: {map50_val:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - mAP@0.5 is often higher than mAP@0.5:0.95 as it uses a single, more lenient IoU threshold
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - Compatible with any pip-installed ultralytics version
    """
    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes = _prepare_detection_data(y_true, y_pred)

    # Define metric calculation function that resamples images
    def map50_metric(image_indices_y_true, image_indices_y_pred):
        """Calculate mAP@0.5 for a subset of images (for bootstrap)."""
        # Use the first argument (both are the same - image indices)
        subset_preds = [predictions[i] for i in image_indices_y_true]
        subset_gts = [ground_truths[i] for i in image_indices_y_true]
        subset_shapes = [image_shapes[i] for i in image_indices_y_true]

        metrics = _calculate_metrics_from_data(subset_preds, subset_gts, subset_shapes)
        return metrics['map50']

    # Calculate mAP@0.5 with optional CI
    if not compute_ci:
        # No CI computation - force plot to False
        plot = False
        all_indices = list(range(len(predictions)))
        return map50_metric(all_indices, all_indices)

    # Extract bootstrap parameters
    n_resamples = kwargs.get('n_resamples', 100)
    random_state = kwargs.get('random_state', 42)

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # Use bootstrap_with_plot helper to reduce code duplication
    return bootstrap_with_plot(
        y_true=image_indices,
        y_pred=image_indices,
        metric_func=map50_metric,
        metric_name="mAP@0.5",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot,
        plot_type="detection"
    )


def yolo_precision(y_true: Union[str, Path],
                   y_pred: List[Any],
                   confidence_level: float = 0.95,
                   method: str = 'bootstrap_percentile',
                   compute_ci: bool = True,
                   plot: bool = False,
                   **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mean precision with confidence interval.

    This function calculates the mean precision across all classes
    with confidence intervals using bootstrap resampling at the image level.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method ('bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'),
        by default 'bootstrap_percentile'
    compute_ci : bool, optional
        If True return the confidence interval as well as the metric score, by default True
    plot : bool, optional
        If True create histogram plot for bootstrap methods, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean precision score and optionally the confidence interval

    Examples
    --------
    >>> from ultralytics import YOLO
    >>> from confidenceinterval import yolo_precision
    >>>
    >>> model = YOLO('yolov8n.pt')
    >>> results = model.predict(source='val-dataset/images/')
    >>>
    >>> # Calculate mean precision with CI
    >>> precision_val, (lower, upper) = yolo_precision(
    ...     y_true='val-dataset',
    ...     y_pred=results,
    ...     method='bootstrap_percentile',
    ...     n_resamples=1000
    ... )
    >>> print(f"Precision: {precision_val:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - Precision = TP / (TP + FP) - measures how many predictions were correct
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - Compatible with any pip-installed ultralytics version
    """
    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes = _prepare_detection_data(y_true, y_pred)

    # Define metric calculation function that resamples images
    def precision_metric(image_indices_y_true, image_indices_y_pred):
        """Calculate precision for a subset of images (for bootstrap)."""
        # Use the first argument (both are the same - image indices)
        subset_preds = [predictions[i] for i in image_indices_y_true]
        subset_gts = [ground_truths[i] for i in image_indices_y_true]
        subset_shapes = [image_shapes[i] for i in image_indices_y_true]

        metrics = _calculate_metrics_from_data(subset_preds, subset_gts, subset_shapes)
        return metrics['precision']

    # Calculate precision with optional CI
    if not compute_ci:
        # No CI computation - force plot to False
        plot = False
        all_indices = list(range(len(predictions)))
        return precision_metric(all_indices, all_indices)

    # Extract bootstrap parameters
    n_resamples = kwargs.get('n_resamples', 100)
    random_state = kwargs.get('random_state', 42)

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # Use bootstrap_with_plot helper to reduce code duplication
    return bootstrap_with_plot(
        y_true=image_indices,
        y_pred=image_indices,
        metric_func=precision_metric,
        metric_name="Precision",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot,
        plot_type="detection"
    )


def yolo_recall(y_true: Union[str, Path],
                y_pred: List[Any],
                confidence_level: float = 0.95,
                method: str = 'bootstrap_percentile',
                compute_ci: bool = True,
                plot: bool = False,
                **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute YOLO mean recall with confidence interval.

    This function calculates the mean recall across all classes
    with confidence intervals using bootstrap resampling at the image level.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method ('bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'),
        by default 'bootstrap_percentile'
    compute_ci : bool, optional
        If True return the confidence interval as well as the metric score, by default True
    plot : bool, optional
        If True create histogram plot for bootstrap methods, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean recall score and optionally the confidence interval

    Examples
    --------
    >>> from ultralytics import YOLO
    >>> from confidenceinterval import yolo_recall
    >>>
    >>> model = YOLO('yolov8n.pt')
    >>> results = model.predict(source='val-dataset/images/')
    >>>
    >>> # Calculate mean recall with CI
    >>> recall_val, (lower, upper) = yolo_recall(
    ...     y_true='val-dataset',
    ...     y_pred=results,
    ...     method='bootstrap_percentile',
    ...     n_resamples=1000
    ... )
    >>> print(f"Recall: {recall_val:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - Recall = TP / (TP + FN) - measures how many ground truth objects were detected
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - Compatible with any pip-installed ultralytics version
    """
    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes = _prepare_detection_data(y_true, y_pred)

    # Define metric calculation function that resamples images
    def recall_metric(image_indices_y_true, image_indices_y_pred):
        """Calculate recall for a subset of images (for bootstrap)."""
        # Use the first argument (both are the same - image indices)
        subset_preds = [predictions[i] for i in image_indices_y_true]
        subset_gts = [ground_truths[i] for i in image_indices_y_true]
        subset_shapes = [image_shapes[i] for i in image_indices_y_true]

        metrics = _calculate_metrics_from_data(subset_preds, subset_gts, subset_shapes)
        return metrics['recall']

    # Calculate recall with optional CI
    if not compute_ci:
        # No CI computation - force plot to False
        plot = False
        all_indices = list(range(len(predictions)))
        return recall_metric(all_indices, all_indices)

    # Extract bootstrap parameters
    n_resamples = kwargs.get('n_resamples', 100)
    random_state = kwargs.get('random_state', 42)

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # Use bootstrap_with_plot helper to reduce code duplication
    return bootstrap_with_plot(
        y_true=image_indices,
        y_pred=image_indices,
        metric_func=recall_metric,
        metric_name="Recall",
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot,
        plot_type="detection"
    )
