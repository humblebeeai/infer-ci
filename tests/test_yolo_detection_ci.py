import time
from confidenceinterval import MetricEvaluator
from ultralytics import YOLO

evaluate = MetricEvaluator()

start = time.time()
model = YOLO('/home/azamjon/Documents/infer/infer-ci/models/yolov8n.pt')

results = model.predict(
    source='/home/azamjon/Documents/infer/infer-ci/custom-dataset/coco128/images',
    # imgsz=960,
    verbose=False,
)
elapsed = time.time() - start

print(f"✓ Predictions completed in {elapsed:.2f}s")
print(f"  - Predicted {len(results)} images")

yolo_map, ci = evaluate.evaluate(
    y_true='/home/azamjon/Documents/infer/infer-ci/custom-dataset/coco128/data.yaml',
    y_pred=results,
    task='detection',
    metric='yolo_map',
    method='bootstrap_percentile',
    plot=True,
    # compute_ci=False
)

print(f"\nResults:")
map_value, (lower, upper) = yolo_map, ci
print(f"  mAP@0.5:0.95 = {map_value:.4f}")
print(f"  95% CI = [{lower:.4f}, {upper:.4f}]")
# print(f"  CI width = {upper - lower:.4f}")