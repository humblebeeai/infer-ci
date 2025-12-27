# Object Detection Metrics Usage Guide

This guide shows how to use the Object detection metrics with confidence intervals.

## Quick Start

```python
from confidenceinterval import MetricEvaluator
from ultralytics import YOLO

# Initialize evaluator
evaluate = MetricEvaluator()

# Load your model and run predictions
model = YOLO('yolov8n.pt')
results = model.predict(source='path/to/val-dataset/images', verbose=False)

# Compute mAP@0.5:0.95 with confidence interval
map_value, (lower, upper) = evaluate.evaluate(
    y_true='path/to/val-dataset',  # Path to dataset folder or data.yaml
    y_pred=results,                 # prediction results
    task='detection',
    metric='map',
    method='bootstrap_percentile',
    n_resamples=1000,
    plot=True                       # Create histogram plot
)

print(f"mAP@0.5:0.95: {map_value:.4f}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
```

## Available Detection Metrics

- `map`: mAP@0.5:0.95 (mean Average Precision across IoU thresholds 0.5:0.05:0.95)
- `map50`: mAP@0.5 (mean Average Precision at IoU threshold 0.5)
- `precision`: Mean precision across all classes
- `recall`: Mean recall across all classes

## Dataset Structure

Your validation dataset should follow to this format:

```
val-dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.txt
    ├── img002.txt
    └── ...
```

Label files should contain one box per line in format:
```
class_id x_center y_center width height
```
Where all values are normalized to [0, 1].

## Per-Class Metrics

Get confidence intervals for each class:

```python
map_value, (lower, upper) = evaluate.evaluate(
    y_true='path/to/dataset',
    y_pred=results,
    task='detection',
    metric='map',
    plot_per_class=True,  # Enable per-class analysis
    n_resamples=1000
)

# Output shows overall + per-class results:
# Per-class mAP@0.5:0.95 with 95% Confidence Intervals:
#      Class                          mAP@0.5:0.95 CI    Support
#      all                            (0.922, 0.949)        1234
#    0 person                         (0.900, 0.969)          45
#    1 car                            (0.789, 0.955)          89
#    ...
```

## Using Different Metrics

```python
# mAP@0.5
map50, ci = evaluate.evaluate(
    y_true='path/to/dataset',
    y_pred=results,
    task='detection',
    metric='map50',
    n_resamples=1000
)

# Precision
precision, ci = evaluate.evaluate(
    y_true='path/to/dataset',
    y_pred=results,
    task='detection',
    metric='precision',
    n_resamples=1000
)

# Recall
recall, ci = evaluate.evaluate(
    y_true='path/to/val-dataset',
    y_pred=results,
    task='detection',
    metric='recall',
    n_resamples=1000
)
```

## Complete Example

📓 **For a complete interactive example with COCO128 dataset, see:** [`notebooks/example_detection.ipynb`](../notebooks/example_detection.ipynb)

The notebook includes:
- Step-by-step setup with COCO128 dataset

## Parameters

### Required
- `y_true`: Path to validation dataset directory or `data.yaml` file
- `y_pred`: List of ultralytics Results objects from `model.predict()`

### Optional
- `confidence_level`: Confidence level (default: 0.95)
- `method`: Bootstrap method (default: 'bootstrap_percentile')
  - `'bootstrap_percentile'`: Percentile method (prefered to use)
  - `'bootstrap_bca'`: Bias-corrected and accelerated (more accurate, slower)
  - `'bootstrap_basic'`: Basic bootstrap
- `n_resamples`: Number of bootstrap resamples (default: 1000)
- `random_state`: Random seed for reproducibility (default: 42)
- `plot`: Create histogram plot for overall metric (default: False)
- `plot_per_class`: Create separate plots for each class (default: False)
- `compute_ci`: Return confidence interval (default: True)

## Notes

- **Bootstrap resampling** is done at the image level (resamples which images to include)
- **No re-inference** is needed; predictions are reused for each bootstrap iteration
- **Per-class analysis** computes all classes in a single bootstrap pass (50x faster than separate)
- **COCO-standard AP** calculation matches ultralytics/official COCO metrics exactly
- Works with any pip-installed `ultralytics` version

## Troubleshooting

### "No matching ground truth found"
- Ensure image filenames match label filenames
- Check that labels are in `dataset/labels/` directory
- Label files should have `.txt` extension

### "No label files found"
- Verify the dataset path is correct
- Ensure the directory structure follows above format

### "Invalid label format"
- Check that label files have 5 columns: `class x_center y_center width height`
- Ensure all values are normalized to [0, 1]
- Verify there are no extra spaces or invalid characters
