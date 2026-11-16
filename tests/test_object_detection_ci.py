"""End-to-end detection test with a real YOLO model on coco128.

Marked ``slow`` and excluded from the default run (see pytest.ini). It is
skipped with a reason when ultralytics is not installed or the coco128
dataset is missing. Run it with::

    python -m pytest -m slow

The fast checks of the detection metrics, which need neither ultralytics
nor a dataset, are in test_detection_synthetic.py.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import infer_ci

COCO128 = Path(__file__).resolve().parents[1] / 'datasets' / 'coco128'

# YOLO runs in a separate process. Other test files import xgboost, and on
# macOS loading PyTorch into the same process crashes (segmentation fault,
# clashing OpenMP runtimes). A child process keeps PyTorch isolated.
_RUN_YOLO = r'''
import json, sys
from ultralytics import YOLO
from infer_ci import MetricEvaluator

dataset = sys.argv[1]
results = YOLO('yolov8n.pt').predict(source=dataset + '/images', verbose=False)
value, (lower, upper), plot_path = MetricEvaluator().evaluate(
    y_true=dataset, y_pred=results, task='detection', metric='map',
    method='bootstrap_percentile', n_resamples=200, plot=True, random_state=42,
)
print(json.dumps({'n_images': len(results), 'value': float(value),
                  'lower': float(lower), 'upper': float(upper), 'plot': str(plot_path)}))
'''


@pytest.mark.slow
def test_yolo_map_on_coco128(tmp_path):
    if importlib.util.find_spec('ultralytics') is None:
        pytest.skip('ultralytics is not installed')
    if not (COCO128 / 'images').is_dir() or not (COCO128 / 'labels').is_dir():
        pytest.skip(f'coco128 dataset not found at {COCO128} (see tests/README.md)')

    # Use the same infer_ci as this test run, and run in a temporary folder
    # so the model download and the plot do not land in the repository.
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(Path(infer_ci.__file__).resolve().parents[1]), env.get('PYTHONPATH', '')])
    run = subprocess.run([sys.executable, '-c', _RUN_YOLO, str(COCO128)],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])

    assert result['n_images'] > 0
    assert 0.0 <= result['lower'] <= result['value'] <= result['upper'] <= 1.0
    # yolov8n gives about 0.49 on coco128; the bound is loose so that
    # ultralytics or torch upgrades do not break the test.
    assert result['value'] > 0.3
    assert (tmp_path / result['plot']).is_file()
