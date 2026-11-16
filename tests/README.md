# Tests

Run the default test suite from the repository root:

```bash
python -m pytest tests/
```

## Slow tests

Tests marked `slow` are excluded by default (`pytest.ini`). At the moment
there is one: `test_object_detection_ci.py`, which runs a real YOLO model on
the coco128 dataset (about 20 seconds). Run it with:

```bash
python -m pytest -m slow
```

It needs:

- `ultralytics` (installs PyTorch): `pip install ultralytics`
- the coco128 dataset in `datasets/coco128/` (with `images/` and `labels/`).
  `datasets/` is ignored by git, so download it once:

  ```bash
  mkdir -p datasets
  curl -L -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
  unzip -q coco128.zip -d datasets && rm coco128.zip
  ```

If either is missing, the test is skipped with the reason.
The model file `yolov8n.pt` is downloaded into a temporary folder on each run.

The detection metrics are also covered by `test_detection_synthetic.py`, which
needs neither `ultralytics` nor a dataset and runs in the default suite.
