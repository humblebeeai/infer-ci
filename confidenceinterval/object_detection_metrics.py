# Confidence intervals for object detection metrics

from typing import List, Tuple, Union, Dict, Any, Optional, Callable
import numpy as np
from pathlib import Path
import yaml
import warnings
import logging

from .visualize import bootstrap_with_plot

# Setup logger
logger = logging.getLogger(__name__)

# Helper Functions (Internal)

def _load_ground_truth(y_true: Union[str, Path]) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Dict[int, str]]:
    """
    Load ground truth boxes from validation dataset directory.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory (contains images/ and labels/ folders)
        or path to data.yaml file

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int]], Dict[int, str]]
        - List of GT boxes per image, each as array [n_boxes, 5] with format [class, x_center, y_center, w, h]
        - List of image shapes (height, width) per image
        - Dictionary mapping class ID to class name

    Raises
    ------
    FileNotFoundError
        If the dataset path, labels directory, or images directory doesn't exist
    ValueError
        If no label files are found or YAML file is invalid
    """
    y_true = Path(y_true)

    # Validate path exists
    if not y_true.exists():
        raise FileNotFoundError(f"Dataset path not found: {y_true}")

    # If y_true is a YAML file, load the path from it
    if y_true.suffix in ['.yaml', '.yml']:
        try:
            with open(y_true, 'r') as f:
                data = yaml.safe_load(f)
                if data is None:
                    raise ValueError(f"YAML file is empty or invalid: {y_true}")

                dataset_path = Path(data.get('path', y_true.parent))
                names = data.get('names', {})

                # Convert list to dict if needed
                if isinstance(names, list):
                    names = {i: name for i, name in enumerate(names)}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {y_true}: {e}")
    else:
        dataset_path = y_true
        names = {}

    # Find labels directory
    labels_dir = dataset_path / 'labels'
    images_dir = dataset_path / 'images'

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_dir}\n"
            f"Expected structure: {dataset_path}/labels/ and {dataset_path}/images/"
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            f"Expected structure: {dataset_path}/labels/ and {dataset_path}/images/"
        )

    # Load all label files
    label_files = sorted(labels_dir.glob('*.txt'))

    # Validate that we have label files
    if len(label_files) == 0:
        raise ValueError(
            f"No label files (*.txt) found in {labels_dir}\n"
            f"Please ensure your dataset has ground truth labels."
        )

    ground_truth_boxes = []
    image_shapes = []
    skipped_labels = []

    for label_file in label_files:
        # First, check if the corresponding image exists
        image_file = images_dir / f"{label_file.stem}.jpg"
        if not image_file.exists():
            image_file = images_dir / f"{label_file.stem}.png"

        if not image_file.exists():
            # Skip this label file if image doesn't exist
            skipped_labels.append(label_file.stem)
            continue

        # Read label file
        if label_file.stat().st_size == 0:
            # Empty file (no objects in image)
            ground_truth_boxes.append(np.zeros((0, 5)))
        else:
            try:
                boxes = np.loadtxt(label_file).reshape(-1, 5)
                # Validate box format
                if boxes.shape[1] != 5:
                    raise ValueError(
                        f"Invalid label format in {label_file.name}: expected 5 columns "
                        f"(class, x_center, y_center, width, height), got {boxes.shape[1]}"
                    )
                ground_truth_boxes.append(boxes)
            except Exception as e:
                raise ValueError(
                    f"Failed to load label file {label_file.name}: {e}\n"
                    f"Expected format: class x_center y_center width height (one box per line)"
                )

        # Get image shape from corresponding image
        from PIL import Image
        img = Image.open(image_file)
        image_shapes.append(img.size[::-1])  # (height, width)

    # Warn if labels were skipped
    if len(skipped_labels) > 0:
        print(f"\n  Skipped {len(skipped_labels)} labels without matching images:")
        for fname in skipped_labels:
            print(f"     - image {fname} not found")
        print()
        

    # Validate we have ground truth
    total_boxes = sum(len(boxes) for boxes in ground_truth_boxes)
    if total_boxes == 0:
        raise ValueError(
            f"No ground truth boxes found across {len(ground_truth_boxes)} label files. "
            f"All label files are empty. Cannot compute detection metrics."
        )

    logger.info(f"Loaded {len(ground_truth_boxes)} label files with {total_boxes} total ground truth boxes")

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

    Raises
    ------
    ValueError
        If y_pred is empty or contains invalid Result objects
    """
    if not y_pred:
        raise ValueError(
            "Predictions list is empty. Please ensure model.predict() returned results."
        )

    predictions = []
    filenames = []

    for idx, result in enumerate(y_pred):
        # Validate Result object has required attributes
        if not hasattr(result, 'path'):
            raise ValueError(
                f"Invalid Result object at index {idx}: missing 'path' attribute. "
                f"Expected ultralytics.engine.results.Results object."
            )
        if not hasattr(result, 'boxes'):
            raise ValueError(
                f"Invalid Result object at index {idx}: missing 'boxes' attribute. "
                f"Expected ultralytics.engine.results.Results object."
            )

        # Get filename from path
        try:
            image_path = Path(result.path)
            filenames.append(image_path.stem)
        except Exception as e:
            raise ValueError(
                f"Invalid path in Result object at index {idx}: {result.path}. Error: {e}"
            )

        if result.boxes is not None and len(result.boxes) > 0:
            try:
                # Extract boxes, confidence, and class
                boxes = result.boxes.xyxy.cpu().numpy()  # [n, 4] - x1y1x2y2 format
                conf = result.boxes.conf.cpu().numpy()    # [n]
                cls = result.boxes.cls.cpu().numpy()      # [n]

                # Concatenate to [n, 6]
                pred_array = np.column_stack([boxes, conf, cls])
                predictions.append(pred_array)
            except Exception as e:
                raise ValueError(
                    f"Failed to extract predictions from Result at index {idx}: {e}"
                )
        else:
            # No detections in this image
            predictions.append(np.zeros((0, 6)))

    logger.info(f"Parsed {len(predictions)} prediction results")

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
                  gt_cls: np.ndarray, eps: float = 1e-16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    eps : float, optional
        Small epsilon value to prevent division by zero, by default 1e-16

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - AP per class and IoU threshold [n_classes, n_iou_thresholds]
        - Precision per class [n_classes]
        - Recall per class [n_classes]

    Raises
    ------
    ValueError
        If ground truth is empty or input arrays have incompatible shapes
    """
    # Validate inputs
    if len(gt_cls) == 0:
        raise ValueError(
            "Ground truth is empty - cannot compute AP. "
            "Please ensure your dataset has ground truth labels."
        )

    if len(tp) != len(conf) or len(tp) != len(pred_cls):
        raise ValueError(
            f"Input array length mismatch: tp={len(tp)}, conf={len(conf)}, pred_cls={len(pred_cls)}. "
            f"All arrays must have the same length."
        )

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
        # Interpolate recall curve
        # Note: Using negative values (-x, -conf[i]) because np.interp requires
        # xp (confidence) to be increasing, but our confidence is sorted in descending order
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision curve
        precision = tpc / (tpc + fpc)
        # Interpolate precision curve (same negative indexing as above)
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
                                 image_shapes: List[Tuple[int, int]],
                                 return_per_class: bool = False) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate all detection metrics from predictions and ground truth.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays per image [n_pred, 6] as [x1, y1, x2, y2, conf, class]
    ground_truths : List[np.ndarray]
        List of GT arrays per image [n_gt, 5] as [class, x_center, y_center, w, h]
    image_shapes : List[Tuple[int, int]]
        List of image shapes (height, width)
    return_per_class : bool, optional
        If True, return per-class metrics in addition to averages, by default False

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        If return_per_class=False:
            Dictionary with metrics: 'map', 'map50', 'precision', 'recall'
        If return_per_class=True:
            Dictionary with:
            - 'map': float - Overall mAP
            - 'map50': float - Overall mAP@0.5
            - 'precision': float - Overall mean precision
            - 'recall': float - Overall mean recall
            - 'ap': np.ndarray [n_classes, n_iou_thresholds] - AP per class
            - 'precision_per_class': np.ndarray [n_classes] - Precision per class
            - 'recall_per_class': np.ndarray [n_classes] - Recall per class
            - 'unique_classes': np.ndarray [n_classes] - Class IDs present

    Raises
    ------
    ValueError
        If input lists have different lengths or are empty
    """
    # Validate inputs
    if not predictions or not ground_truths or not image_shapes:
        raise ValueError(
            f"Input lists cannot be empty: predictions={len(predictions)}, "
            f"ground_truths={len(ground_truths)}, image_shapes={len(image_shapes)}"
        )

    if not (len(predictions) == len(ground_truths) == len(image_shapes)):
        raise ValueError(
            f"Input lists must have the same length: predictions={len(predictions)}, "
            f"ground_truths={len(ground_truths)}, image_shapes={len(image_shapes)}"
        )

    # Convert GT to xyxy
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

        # Get unique classes from ground truth
        unique_classes = np.unique(gt_cls)
    else:
        map_value = 0.0
        map50_value = 0.0
        precision_mean = 0.0
        recall_mean = 0.0
        unique_classes = np.array([])

    # Return per-class metrics if requested
    if return_per_class:
        return {
            'map': map_value,
            'map50': map50_value,
            'precision': precision_mean,
            'recall': recall_mean,
            'ap': ap,  # [n_classes, n_iou_thresholds]
            'precision_per_class': precision,  # [n_classes]
            'recall_per_class': recall,  # [n_classes]
            'unique_classes': unique_classes  # [n_classes]
        }
    else:
        return {
            'map': map_value,
            'map50': map50_value,
            'precision': precision_mean,
            'recall': recall_mean
        }


def _compute_percentile_ci_from_samples(samples: np.ndarray,
                                        original_value: float,
                                        confidence_level: float) -> Tuple[float, Tuple[float, float]]:
    """
    Compute percentile-based confidence interval from bootstrap samples.

    This helper uses the percentile method which is simple and robust for per-class metrics.
    For overall metrics, use bootstrap_with_plot which supports all methods (percentile, basic, BCA).

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap samples
    original_value : float
        The original metric value computed on the full dataset (not bootstrap mean)
    confidence_level : float
        Confidence level (e.g., 0.95)

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        (metric_value, (lower_bound, upper_bound))

    Notes
    -----
    This function uses the same percentile calculation as scipy.stats.bootstrap for consistency.
    """
    metric_value = original_value

    # Calculate percentiles (same logic as scipy.stats.bootstrap)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci = (np.percentile(samples, lower_percentile),
          np.percentile(samples, upper_percentile))

    return metric_value, ci


def _compute_per_class_ci(predictions: List[np.ndarray],
                                    ground_truths: List[np.ndarray],
                                    image_shapes: List[Tuple[int, int]],
                                    class_names: Dict[int, str],
                                    metric_name: str,
                                    metric_extractor: Callable,
                                    confidence_level: float,
                                    method: str,
                                    n_resamples: int,
                                    random_state: int = 42,
                                    plot: bool = True) -> Tuple[float, Tuple[float, float]]:
    """
    Efficiently compute overall AND per-class confidence intervals in a single bootstrap pass.

    This function runs bootstrap ONCE and captures both overall and per-class metrics
    in each iteration, making it ~50x faster than computing them separately.

    Parameters
    ----------
    predictions : List[np.ndarray]
        List of prediction arrays per image
    ground_truths : List[np.ndarray]
        List of ground truth arrays per image
    image_shapes : List[Tuple[int, int]]
        List of image shapes (height, width)
    class_names : Dict[int, str]
        Dictionary mapping class ID to class name
    metric_name : str
        Name of the metric (e.g., "mAP@0.5:0.95", "Precision")
    metric_extractor : callable
        Function that extracts the metric value for a specific class from metrics dict.
        Signature: metric_extractor(metrics: dict, class_idx: int) -> float
    confidence_level : float
        The confidence interval level
    method : str
        The bootstrap method
    n_resamples : int
        Number of bootstrap resamples
    random_state : Any
        Random state for reproducibility
    plot : bool
        Whether to create plots for each class

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        - overall_value: Overall metric value
        - overall_ci: Overall confidence interval (lower, upper)

    Notes
    -----
    Per-class results are printed to console and saved as plots.
    This maintains backward compatibility with the standard return format.
    """
    # Get all unique classes from ground truth
    all_gt_cls = np.concatenate([gt[:, 0] for gt in ground_truths if len(gt) > 0])
    unique_classes = np.unique(all_gt_cls).astype(int)

    # First, compute original metric values on full dataset
    original_metrics = _calculate_metrics_from_data(predictions, ground_truths, image_shapes,
                                                    return_per_class=True)

    # Extract overall original value
    if "mAP@0.5:0.95" in metric_name:
        original_overall = original_metrics['map']
    elif "mAP@0.5" in metric_name:
        original_overall = original_metrics['map50']
    elif "Precision" in metric_name:
        original_overall = original_metrics['precision']
    elif "Recall" in metric_name:
        original_overall = original_metrics['recall']

    # Extract per-class original values
    original_per_class = {}
    for class_id in unique_classes:
        class_indices = np.where(original_metrics['unique_classes'] == class_id)[0]
        if len(class_indices) > 0:
            class_idx = class_indices[0]
            original_per_class[class_id] = metric_extractor(original_metrics, class_idx)
        else:
            original_per_class[class_id] = 0.0

    # Storage for bootstrap samples
    n_images = len(predictions)
    overall_samples = []
    per_class_samples = {class_id: [] for class_id in unique_classes}

    # Run bootstrap iterations
    rng = np.random.default_rng(random_state)

    print(f"Bootstrap resampling: {n_resamples} iterations...")
    from tqdm import tqdm
    for _ in tqdm(range(n_resamples), desc="Bootstrap CI"):
        # Resample image indices
        resampled_indices = rng.choice(n_images, size=n_images, replace=True)

        # Get subset of data
        subset_preds = [predictions[i] for i in resampled_indices]
        subset_gts = [ground_truths[i] for i in resampled_indices]
        subset_shapes = [image_shapes[i] for i in resampled_indices]

        # Compute metrics with per-class breakdown
        metrics = _calculate_metrics_from_data(subset_preds, subset_gts, subset_shapes,
                                               return_per_class=True)

        # Store overall metric (extract from metrics dict based on metric_name)
        if "mAP@0.5:0.95" in metric_name:
            overall_samples.append(metrics['map'])
        elif "mAP@0.5" in metric_name:
            overall_samples.append(metrics['map50'])
        elif "Precision" in metric_name:
            overall_samples.append(metrics['precision'])
        elif "Recall" in metric_name:
            overall_samples.append(metrics['recall'])

        # Store per-class metrics
        for class_id in unique_classes:
            class_indices = np.where(metrics['unique_classes'] == class_id)[0]
            if len(class_indices) == 0:
                per_class_samples[class_id].append(0.0)
            else:
                class_idx = class_indices[0]
                per_class_samples[class_id].append(metric_extractor(metrics, class_idx))

    # Convert to arrays
    overall_samples = np.array(overall_samples)

    overall_value, overall_ci = _compute_percentile_ci_from_samples(
        overall_samples, original_overall, confidence_level
    )

    # Count support (number of ground truth instances) per class
    support_counts = {}
    for gt in ground_truths:
        for class_id in gt[:, 0]:
            class_id = int(class_id)
            support_counts[class_id] = support_counts.get(class_id, 0) + 1

    # Compute per-class CIs and optionally plot
    per_class_results = {}
    table_data = []

    print(f"Saving plots ...") if plot else None

    for class_id in unique_classes:
        class_name = class_names.get(class_id, f"class_{class_id}")
        samples = np.array(per_class_samples[class_id])

        class_value, class_ci = _compute_percentile_ci_from_samples(
            samples, original_per_class[class_id], confidence_level
        )

        per_class_results[class_id] = (class_value, class_ci)
        support = support_counts.get(class_id, 0)

        # Add to table data
        table_data.append({
            'class_id': class_id,
            'class_name': class_name,
            'value': class_value,
            'ci': class_ci,
            'support': support
        })

        # Plot if requested (generate all plots)        
        if plot:
            from .visualize import create_bootstrap_histogram_plot
            plot_name = f"{metric_name} - class_{class_id}_{class_name}"
            create_bootstrap_histogram_plot(
                samples, class_value, class_ci,
                plot_name, method, confidence_level,
                plot_type="detection"
            )

    # Print results as formatted table
    print(f"\nPer-class {metric_name} with {int(confidence_level*100)}% Confidence Intervals:")

    total_support = sum(support_counts.values())

    # Header
    print(f"{'':>4} {'Class':<30} {metric_name+' CI':<25} {'Support':>8}")

    # Overall row
    print(f"{'':>4} {'all':<30} ({overall_ci[0]:.3f}, {overall_ci[1]:.3f}){'':<13} {total_support:>6}")

    # Per-class rows
    for row in table_data:
        class_label = f"{row['class_id']}  {row['class_name']}"
        ci_str = f"({row['ci'][0]:.3f}, {row['ci'][1]:.3f})"
        print(f"{row['class_id']:>4} {row['class_name']:<30} {ci_str:<25} {row['support']:>8}")

    if plot:
        print(f"\n✓ Saved per-class plots to results/")
    print()

    return overall_value, overall_ci


def _prepare_detection_data(y_true: Union[str, Path],
                            y_pred: List[Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], Dict[int, str]]:
    """
    Prepare detection data by loading ground truth and matching to predictions.

    Parameters
    ----------
    y_true : Union[str, Path]
        Path to validation dataset directory or YAML file
    y_pred : List[Any]
        List of ultralytics Results objects from model.predict()

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], Dict[int, str]]
        - matched_predictions: List of prediction arrays per image
        - ground_truths: List of ground truth arrays per image
        - image_shapes: List of image shapes (height, width)
        - class_names: Dictionary mapping class ID to class name
    """
    # Load all ground truth
    all_ground_truths, all_image_shapes, class_names = _load_ground_truth(y_true)

    # Get class names from predictions if not available from YAML
    if not class_names and len(y_pred) > 0:
        # Ultralytics Results objects have a 'names' attribute with class names
        if hasattr(y_pred[0], 'names'):
            class_names = y_pred[0].names

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
        print(f"  Skipped {len(skipped_files)} predictions without matching ground truth labels:")
        for fname in skipped_files:
            print(f"     - label {fname}.txt file not found")
        print()

    if len(matched_predictions) == 0:
        raise ValueError(
            "No matching ground truth found for any predictions. "
            "Please check that your ground truth labels match the predicted image filenames."
        )

    return matched_predictions, ground_truths, image_shapes, class_names

# Main Metric Functions

def map(y_true: Union[str, Path],
             y_pred: List[Any],
             confidence_level: float = 0.95,
             method: str = 'bootstrap_percentile',
             compute_ci: bool = True,
             plot: bool = False,
             plot_per_class: bool = False,
             random_state: int = 42,
             **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute mAP@0.5:0.95 with confidence interval.

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
        If True create histogram plot for overall metric, by default False
    plot_per_class : bool, optional
        If True create separate histogram plots for each class, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP@0.5:0.95 score and optionally the confidence interval

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - This is the standard approach for object detection metrics
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - plot_per_class creates separate plots for each class showing performance distribution
    - Compatible with any pip-installed ultralytics version

    Raises
    ------
    ValueError
        If plot_per_class=True but compute_ci=False
    """
    # Parameter validation
    if plot_per_class and not compute_ci:
        raise ValueError("plot_per_class requires compute_ci=True")

    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes, class_names = _prepare_detection_data(y_true, y_pred)

    # Define metric calculation function that resamples images
    def map_metric(image_indices_y_true, image_indices_y_pred):
        """Calculate mAP for a subset of images (for bootstrap).

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
    n_resamples = kwargs.get('n_resamples', 1000)  # Higher default for production

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # EFFICIENT per-class computation: Run bootstrap ONCE for both overall and per-class
    if plot_per_class:
        return _compute_per_class_ci(
            predictions=predictions,
            ground_truths=ground_truths,
            image_shapes=image_shapes,
            class_names=class_names,
            metric_name="mAP@0.5:0.95",
            metric_extractor=lambda metrics, idx: metrics['ap'][idx].mean(),  # Mean over IoU thresholds
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            plot=plot
        )

    # Use bootstrap_with_plot helper for overall metric only
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


def map50(y_true: Union[str, Path],
               y_pred: List[Any],
               confidence_level: float = 0.95,
               method: str = 'bootstrap_percentile',
               compute_ci: bool = True,
               plot: bool = False,
               plot_per_class: bool = False,
               random_state: int = 42,
               **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute mAP@0.5 with confidence interval.

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
        If True create histogram plot for overall metric, by default False
    plot_per_class : bool, optional
        If True create separate histogram plots for each class, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mAP@0.5 score and optionally the confidence interval

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - mAP@0.5 is often higher than mAP@0.5:0.95 as it uses a single, more lenient IoU threshold
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - plot_per_class creates separate plots for each class showing performance distribution
    - Compatible with any pip-installed ultralytics version

    Raises
    ------
    ValueError
        If plot_per_class=True but compute_ci=False
    """
    # Parameter validation
    if plot_per_class and not compute_ci:
        raise ValueError("plot_per_class requires compute_ci=True")

    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes, class_names = _prepare_detection_data(y_true, y_pred)

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
    n_resamples = kwargs.get('n_resamples', 1000)  # Higher default for production

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # EFFICIENT per-class computation: Run bootstrap ONCE for both overall and per-class
    if plot_per_class:
        return _compute_per_class_ci(
            predictions=predictions,
            ground_truths=ground_truths,
            image_shapes=image_shapes,
            class_names=class_names,
            metric_name="mAP@0.5",
            metric_extractor=lambda metrics, idx: metrics['ap'][idx, 0],  # First IoU threshold
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            plot=plot
        )

    # Use bootstrap_with_plot helper for overall metric only
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


def precision(y_true: Union[str, Path],
                   y_pred: List[Any],
                   confidence_level: float = 0.95,
                   method: str = 'bootstrap_percentile',
                   compute_ci: bool = True,
                   plot: bool = False,
                   plot_per_class: bool = False,
                   random_state: int = 42,
                   **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute mean precision with confidence interval.

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
        If True create histogram plot for overall metric, by default False
    plot_per_class : bool, optional
        If True create separate histogram plots for each class, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean precision score and optionally the confidence interval

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - Precision = TP / (TP + FP) - measures how many predictions were correct
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - plot_per_class creates separate plots for each class showing performance distribution
    - Compatible with any pip-installed ultralytics version

    Raises
    ------
    ValueError
        If plot_per_class=True but compute_ci=False
    """
    # Parameter validation
    if plot_per_class and not compute_ci:
        raise ValueError("plot_per_class requires compute_ci=True")

    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes, class_names = _prepare_detection_data(y_true, y_pred)

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
    n_resamples = kwargs.get('n_resamples', 1000)  # Higher default for production

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # EFFICIENT per-class computation: Run bootstrap ONCE for both overall and per-class
    if plot_per_class:
        return _compute_per_class_ci(
            predictions=predictions,
            ground_truths=ground_truths,
            image_shapes=image_shapes,
            class_names=class_names,
            metric_name="Precision",
            metric_extractor=lambda metrics, idx: metrics['precision_per_class'][idx],
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            plot=plot
        )

    # Use bootstrap_with_plot helper for overall metric only
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


def recall(y_true: Union[str, Path],
                y_pred: List[Any],
                confidence_level: float = 0.95,
                method: str = 'bootstrap_percentile',
                compute_ci: bool = True,
                plot: bool = False,
                plot_per_class: bool = False,
                random_state: int = 42,
                **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute mean recall with confidence interval.

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
        If True create histogram plot for overall metric, by default False
    plot_per_class : bool, optional
        If True create separate histogram plots for each class, by default False
    **kwargs
        Additional arguments passed to bootstrap methods (e.g., n_resamples, random_state)

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The mean recall score and optionally the confidence interval

    Notes
    -----
    - Bootstrap resampling is done at the image level (resamples which images to include)
    - Recall = TP / (TP + FN) - measures how many ground truth objects were detected
    - No re-inference is needed; predictions are reused for each bootstrap sample
    - Number of classes is automatically inferred from the ground truth data
    - plot_per_class creates separate plots for each class showing performance distribution
    - Compatible with any pip-installed ultralytics version

    Raises
    ------
    ValueError
        If plot_per_class=True but compute_ci=False
    """
    # Parameter validation
    if plot_per_class and not compute_ci:
        raise ValueError("plot_per_class requires compute_ci=True")

    # Prepare detection data (eliminates code duplication)
    predictions, ground_truths, image_shapes, class_names = _prepare_detection_data(y_true, y_pred)

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
    n_resamples = kwargs.get('n_resamples', 1000)  # Higher default for production

    # Create image indices array for bootstrap resampling
    image_indices = np.arange(len(predictions))

    # EFFICIENT per-class computation: Run bootstrap ONCE for both overall and per-class
    if plot_per_class:
        return _compute_per_class_ci(
            predictions=predictions,
            ground_truths=ground_truths,
            image_shapes=image_shapes,
            class_names=class_names,
            metric_name="Recall",
            metric_extractor=lambda metrics, idx: metrics['recall_per_class'][idx],
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            plot=plot
        )

    # Use bootstrap_with_plot helper for overall metric only
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
