import time
from confidenceinterval import MetricEvaluator
from ultralytics import YOLO

evaluate = MetricEvaluator()

print('\nStarting YOLOv8 detection')
print('=============================')

start = time.time()
model = YOLO('/home/azamjon/Documents/infer/infer-ci/models/pepsi_product_model_v4_yolov8x.pt')

results = model.predict(
    source='/home/azamjon/Documents/infer/infer-ci/custom-dataset/gws-val-dataset/images',
    verbose=False,
)
elapsed = time.time() - start

print(f"✓ Predictions completed in {elapsed:.2f}s")
print(f"  - Predicted {len(results)} images")

# Test with plot_per_class parameter
yolo_map_pc, ci_pc = evaluate.evaluate(
    y_true='/home/azamjon/Documents/infer/infer-ci/custom-dataset/gws-val-dataset',
    y_pred=results,
    task='detection',
    metric='yolo_map',
    method='bootstrap_percentile',
    # plot_per_class=True,  # This creates separate plots for each class!
    n_resamples=100,  # Using fewer resamples for testing
    plot=True,
)

print(f"\nOverall Results:")
map_value_pc, (lower_pc, upper_pc) = yolo_map_pc, ci_pc
print(f"  mAP@0.5:0.95 = {map_value_pc:.4f}")
print(f"  95% CI = [{lower_pc:.4f}, {upper_pc:.4f}]")
print(f"\n✓ Per-class plots saved to results/ directory")