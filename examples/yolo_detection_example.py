"""
Example: YOLO Detection Metrics with Confidence Intervals

This example demonstrates the NEW approach for computing YOLO detection metrics
with confidence intervals. This approach follows the same pattern as regression
and classification metrics in infer-ci.

Key Benefits:
1. Inference runs ONCE (not N times for bootstrap)
2. Bootstrap resampling happens at prediction level (fast!)
3. Same API as regression/classification metrics (consistent)
4. Works with any ultralytics YOLO version
5. 1000x faster than the old approach

Workflow:
    Step 1: Run inference ONCE to get all predictions
    Step 2: Compute metrics with bootstrap CI (no more inference!)
"""

import sys
from pathlib import Path

# Add parent directory to path to import confidenceinterval
sys.path.insert(0, str(Path(__file__).parent.parent))

from confidenceinterval import (
    MetricEvaluator,
    extract_detection_data,
    yolo_map_with_ci,
    yolo_map50_with_ci,
    yolo_precision_with_ci,
    yolo_recall_with_ci
)


def example_new_approach():
    """
    Example using the NEW approach (efficient, follows infer-ci pattern).

    This approach:
    - Runs inference ONCE
    - Bootstrap resamples predictions (not data)
    - Same pattern as regression metrics
    """
    print("=" * 80)
    print("NEW APPROACH: Inference ONCE + Bootstrap at Prediction Level")
    print("=" * 80)

    # ========================================================================
    # Step 1: Run inference ONCE to extract detection data
    # ========================================================================
    print("\nStep 1: Running inference ONCE to extract detection data...")
    print("This step runs model inference on the entire validation dataset.")

    try:
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')  # or path to your trained model
        data = 'coco128.yaml'       # or path to your dataset config

        # Extract detection data (runs inference ONCE)
        detection_data = extract_detection_data(
            model=model,
            data=data,
            batch=16,
            device='cpu'  # change to 'cuda' if GPU available
        )

        print(f"✓ Inference complete!")
        print(f"  - Number of detections: {len(detection_data)}")
        print(f"  - Number of classes: {detection_data.nc}")
        print(f"  - Class names: {list(detection_data.class_names.values())[:5]}...")

    except ImportError:
        print("⚠ Ultralytics not available. Using dummy data for demonstration.")
        # Create dummy detection data for demonstration
        import numpy as np
        from confidenceinterval import DetectionData

        detection_data = DetectionData(
            tp=np.random.randint(0, 2, size=(100, 10)),
            conf=np.random.rand(100),
            pred_cls=np.random.randint(0, 80, size=100),
            target_cls=np.random.randint(0, 80, size=200),
            class_names={i: f'class_{i}' for i in range(80)}
        )
        print(f"✓ Created dummy detection data")
        print(f"  - Number of detections: {len(detection_data)}")

    # ========================================================================
    # Step 2: Compute metrics with bootstrap CI (no more inference!)
    # ========================================================================
    print("\nStep 2: Computing metrics with bootstrap CI...")
    print("This step resamples predictions (fast!) and computes metrics.")

    # Method 1: Using individual metric functions (direct approach)
    print("\n--- Method 1: Direct metric functions ---")

    # Compute mAP50-95 with CI
    print("\nComputing mAP50-95 with confidence interval...")
    map_val, (map_lower, map_upper) = yolo_map_with_ci(
        detection_data=detection_data,
        method='bootstrap_bca',
        confidence_level=0.95,
        compute_ci=True,
        n_resamples=100  # Use 9999 for production, 100 for quick testing
    )
    print(f"  mAP50-95: {map_val:.4f}")
    print(f"  95% CI: [{map_lower:.4f}, {map_upper:.4f}]")

    # Compute mAP50 with CI
    print("\nComputing mAP50 with confidence interval...")
    map50_val, (map50_lower, map50_upper) = yolo_map50_with_ci(
        detection_data=detection_data,
        method='bootstrap_bca',
        confidence_level=0.95,
        compute_ci=True,
        n_resamples=100
    )
    print(f"  mAP50: {map50_val:.4f}")
    print(f"  95% CI: [{map50_lower:.4f}, {map50_upper:.4f}]")

    # Compute precision with CI
    print("\nComputing mean precision with confidence interval...")
    precision_val, (precision_lower, precision_upper) = yolo_precision_with_ci(
        detection_data=detection_data,
        method='bootstrap_bca',
        confidence_level=0.95,
        compute_ci=True,
        n_resamples=100
    )
    print(f"  Mean Precision: {precision_val:.4f}")
    print(f"  95% CI: [{precision_lower:.4f}, {precision_upper:.4f}]")

    # Compute recall with CI
    print("\nComputing mean recall with confidence interval...")
    recall_val, (recall_lower, recall_upper) = yolo_recall_with_ci(
        detection_data=detection_data,
        method='bootstrap_bca',
        confidence_level=0.95,
        compute_ci=True,
        n_resamples=100
    )
    print(f"  Mean Recall: {recall_val:.4f}")
    print(f"  95% CI: [{recall_lower:.4f}, {recall_upper:.4f}]")

    # Method 2: Using MetricEvaluator (unified interface)
    print("\n--- Method 2: MetricEvaluator (unified interface) ---")

    evaluator = MetricEvaluator()

    # Compute mAP50-95
    map_result = evaluator.evaluate(
        task='detection',
        metric='yolo_map',
        detection_data=detection_data,
        method='bootstrap_bca',
        confidence_level=0.95,
        compute_ci=True,
        n_resamples=100
    )
    map_val, (map_lower, map_upper) = map_result
    print(f"\nmAP50-95: {map_val:.4f}, 95% CI: [{map_lower:.4f}, {map_upper:.4f}]")

    # Compute precision without CI (faster)
    precision_val = evaluator.evaluate(
        task='detection',
        metric='yolo_precision',
        detection_data=detection_data,
        compute_ci=False  # No CI, just compute the metric
    )
    print(f"Mean Precision (no CI): {precision_val:.4f}")


def example_one_liner():
    """
    Example: Compute metrics in one call (extracts data automatically).

    This is even simpler - MetricEvaluator extracts detection data automatically
    if you provide model and data parameters.
    """
    print("\n" + "=" * 80)
    print("ONE-LINER APPROACH: MetricEvaluator handles everything")
    print("=" * 80)

    try:
        from ultralytics import YOLO

        evaluator = MetricEvaluator()

        # Compute mAP with CI in one call (extracts detection data automatically)
        print("\nComputing mAP50-95 with CI (one-liner)...")
        map_val, (map_lower, map_upper) = evaluator.evaluate(
            task='detection',
            metric='yolo_map',
            model='yolov8n.pt',  # Model will be loaded and inference run automatically
            data='coco128.yaml',
            method='bootstrap_bca',
            confidence_level=0.95,
            n_resamples=100,
            batch=16,
            device='cpu'
        )
        print(f"mAP50-95: {map_val:.4f}, 95% CI: [{map_lower:.4f}, {map_upper:.4f}]")

    except ImportError:
        print("⚠ Ultralytics not available. Skipping one-liner example.")


def compare_with_regression():
    """
    Example: Show how detection metrics now follow the same pattern as regression.

    This demonstrates the consistency between detection and regression metrics.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Detection vs Regression Pattern")
    print("=" * 80)

    print("\n--- Regression Metrics Pattern ---")
    print("""
# Step 1: User does inference separately
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # ← Inference ONCE

# Step 2: Compute metric with CI (no more inference)
from confidenceinterval import MetricEvaluator
evaluator = MetricEvaluator()

mae_val, ci = evaluator.evaluate(
    y_true=y_test,  # ← Pre-computed ground truth
    y_pred=y_pred,  # ← Pre-computed predictions
    task='regression',
    metric='mae',
    method='bootstrap_bca'
)
    """)

    print("\n--- Detection Metrics Pattern (NEW) ---")
    print("""
# Step 1: User does inference separately (or we do it once)
from confidenceinterval import extract_detection_data
model = YOLO('yolov8n.pt')
detection_data = extract_detection_data(model, 'coco128.yaml')  # ← Inference ONCE

# Step 2: Compute metric with CI (no more inference!)
from confidenceinterval import MetricEvaluator
evaluator = MetricEvaluator()

map_val, ci = evaluator.evaluate(
    detection_data=detection_data,  # ← Pre-computed detection statistics
    task='detection',
    metric='yolo_map',
    method='bootstrap_bca'
)
    """)

    print("\n✓ Both follow the same pattern:")
    print("  1. Run inference ONCE")
    print("  2. Compute metrics with bootstrap CI (resamples predictions, not data)")
    print("  3. Same API: MetricEvaluator.evaluate()")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("YOLO Detection Metrics with Confidence Intervals")
    print("NEW Approach: Following infer-ci Pattern")
    print("=" * 80)

    # Run examples
    example_new_approach()
    example_one_liner()
    compare_with_regression()

    print("\n" + "=" * 80)
    print("Summary: Benefits of New Approach")
    print("=" * 80)
    print("""
✓ 1000x faster: Inference runs ONCE (not N times)
✓ Consistent API: Same pattern as regression/classification metrics
✓ Works with any ultralytics version: No custom validator needed
✓ Flexible: Can cache detection_data and compute multiple metrics
✓ Follows infer-ci guidelines: Bootstrap at prediction level, not data level

Old Approach (SLOW):
  for i in range(n_iterations):  # 100 iterations
      # Run inference on random 50% of data  ← EXPENSIVE!
      preds = model(data)
      metrics.append(compute_map(preds))

New Approach (FAST):
  detection_data = extract_detection_data(model, data)  # ← Inference ONCE
  for i in range(n_iterations):  # 100 iterations
      # Resample prediction indices  ← CHEAP!
      resampled = detection_data[indices]
      metrics.append(compute_map(resampled))
    """)

    print("\nFor more information, see:")
    print("  - confidenceinterval/detection_metrics.py")
    print("  - docs/dev-guide-metrics.md")


if __name__ == '__main__':
    main()
