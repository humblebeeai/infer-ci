import time
from infer_ci import MetricEvaluator
from ultralytics import YOLO

evaluate = MetricEvaluator()

print('\nStarting detection')
print('=============================')

start = time.time()
model = YOLO('yolov8n.pt')

results = model.predict(
    source='/home/azamjon/Documents/infer/infer-ci/datasets/coco128/images',
    verbose=False,
)
elapsed = time.time() - start

print(f"✓ Predictions completed in {elapsed:.2f}s")
print(f"  - Predicted {len(results)} images")

metric_value, ci, filepath = evaluate.evaluate(
    y_true='/home/azamjon/Documents/infer/infer-ci/datasets/coco128',
    y_pred=results,
    task='detection',
    metric='map',
    method='bootstrap_percentile',
    # plot_per_class=True,  # This creates separate results for each class!
    n_resamples=1000,
    plot=True,
    random_state=42,  # Fixed seed for reproducible results
)

print(f"\nOverall Results:")
map_value_pc, (lower_pc, upper_pc) = metric_value, ci
print(f"  mAP@0.5:0.95 = {map_value_pc:.4f}")
print(f"  95% CI = [{lower_pc:.4f}, {upper_pc:.4f}]")
print(f"\n✓ Per-class plots saved to results/ directory")
