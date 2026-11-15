"""Detection metrics on a tiny synthetic dataset.

These tests need no ultralytics, torch, model download or real dataset.
The detection code only reads ``result.path`` and
``result.boxes.xyxy / conf / cls`` (via ``.cpu().numpy()``) from each
prediction, so small stand-in objects are enough. The images are blank PNGs,
because ``_load_ground_truth`` only reads their size.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from infer_ci import evaluate_metric, map, map50, precision, recall

IMAGE_WIDTH, IMAGE_HEIGHT = 100, 200
N_IMAGES = 6
BOOTSTRAP = dict(method='bootstrap_percentile', n_resamples=200, random_state=0)

# The YOLO-style AP (101-point COCO interpolation) tops out at 0.995.
PERFECT_AP = 0.995


class _Array:
    """Stands in for a torch tensor: supports .cpu().numpy()."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Array(np.reshape(xyxy, (-1, 4)))
        self.conf = _Array(conf)
        self.cls = _Array(cls)

    def __len__(self):
        return len(self.conf.numpy())


class _Result:
    """Stands in for ultralytics.engine.results.Results."""

    def __init__(self, path, xyxy, conf, cls):
        self.path = str(path)
        self.boxes = _Boxes(xyxy, conf, cls)


@pytest.fixture
def dataset(tmp_path):
    """Six images, one 40x60 px box each, classes alternating 0 and 1.

    Returns the dataset folder and, per image, (class, box in pixel xyxy).
    """
    root = tmp_path / 'dataset'
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir()
    box_w, box_h = 40, 60
    ground_truth = {}
    for i in range(N_IMAGES):
        name = f'img{i}'
        Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT)).save(root / 'images' / f'{name}.png')
        cls = i % 2
        # YOLO label format: class x_center y_center width height (normalised)
        (root / 'labels' / f'{name}.txt').write_text(
            f'{cls} 0.5 0.5 {box_w / IMAGE_WIDTH} {box_h / IMAGE_HEIGHT}\n')
        cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
        ground_truth[name] = (cls, [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2])
    return root, ground_truth


def _perfect_predictions(root, ground_truth):
    return [_Result(root / 'images' / f'{name}.png', box, [0.9], [cls])
            for name, (cls, box) in ground_truth.items()]


def _imperfect_predictions(root, ground_truth):
    """img0 missed, img1 predicted with the wrong class, img2 has an extra box.

    That leaves 4 of the 6 ground-truth boxes found, so recall is 4/6.
    """
    predictions = []
    for name, (cls, box) in ground_truth.items():
        if name == 'img0':
            xyxy, conf, classes = np.zeros((0, 4)), [], []
        elif name == 'img1':
            xyxy, conf, classes = [box], [0.9], [1 - cls]
        elif name == 'img2':
            xyxy, conf, classes = [box, [0, 0, 30, 30]], [0.9, 0.8], [cls, cls]
        else:
            xyxy, conf, classes = [box], [0.9], [cls]
        predictions.append(_Result(root / 'images' / f'{name}.png', xyxy, conf, classes))
    return predictions


@pytest.mark.parametrize('metric, expected', [
    (map, PERFECT_AP),
    (map50, PERFECT_AP),
    (precision, 1.0),
    (recall, 1.0),
])
def test_perfect_predictions(dataset, metric, expected):
    root, ground_truth = dataset
    value, (lower, upper) = metric(str(root), _perfect_predictions(root, ground_truth), **BOOTSTRAP)
    assert value == pytest.approx(expected)
    assert lower == pytest.approx(expected)
    assert upper == pytest.approx(expected)


@pytest.mark.parametrize('metric', [map, map50, precision, recall])
def test_imperfect_predictions_interval(dataset, metric):
    root, ground_truth = dataset
    value, (lower, upper) = metric(str(root), _imperfect_predictions(root, ground_truth), **BOOTSTRAP)
    assert 0.0 <= lower <= value <= upper <= 1.0
    assert value < 1.0
    assert lower < upper


def test_imperfect_recall_value(dataset):
    root, ground_truth = dataset
    value = recall(str(root), _imperfect_predictions(root, ground_truth), compute_ci=False)
    assert value == pytest.approx(4 / 6)


def test_map_not_above_map50(dataset):
    # mAP@0.5:0.95 averages over stricter IoU thresholds, so it cannot exceed mAP@0.5.
    root, ground_truth = dataset
    predictions = _imperfect_predictions(root, ground_truth)
    assert map(str(root), predictions, compute_ci=False) <= map50(str(root), predictions, compute_ci=False)


@pytest.mark.parametrize('metric_name, metric', [
    ('map', map), ('map50', map50), ('precision', precision), ('recall', recall),
])
def test_unified_interface_matches_direct_call(dataset, metric_name, metric):
    root, ground_truth = dataset
    predictions = _imperfect_predictions(root, ground_truth)
    unified = evaluate_metric(str(root), predictions, task='detection', metric=metric_name, **BOOTSTRAP)
    direct = metric(str(root), predictions, **BOOTSTRAP)
    assert unified[0] == pytest.approx(direct[0])
    assert unified[1] == pytest.approx(direct[1])


def test_same_random_state_gives_same_interval(dataset):
    root, ground_truth = dataset
    predictions = _imperfect_predictions(root, ground_truth)
    first = map50(str(root), predictions, **BOOTSTRAP)
    second = map50(str(root), predictions, **BOOTSTRAP)
    assert first[1] == pytest.approx(second[1])


def test_empty_predictions_raise(dataset):
    root, _ = dataset
    with pytest.raises(ValueError, match='empty'):
        map(str(root), [], compute_ci=False)


def test_missing_labels_folder_raises(tmp_path):
    (tmp_path / 'images').mkdir()
    with pytest.raises(FileNotFoundError, match='Labels directory'):
        map(str(tmp_path), [_Result(tmp_path / 'images' / 'a.png', np.zeros((0, 4)), [], [])],
            compute_ci=False)
