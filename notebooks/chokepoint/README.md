# ChokePoint face-recognition attendance

This folder evaluates a face-recognition attendance system on the
[ChokePoint dataset](https://zenodo.org/records/815657) with `infer-ci`.

| File | What it does |
|---|---|
| `chokepoint_data.py` | Downloads ChokePoint archives and loads the ground truth (INF-48) |
| `predict_insightface.py` | Enrols 25 people and writes face predictions as CSV (INF-49) |
| `attendance_metrics.py` | Turns predictions into attendance records and computes the metrics (INF-50) |
| `chokepoint_attendance.ipynb` | Runs the whole example from start to end |

## What to install

Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # infer-ci and its dependencies
pip install matplotlib jupyter
```

To make the predictions yourself (optional), also install:

```bash
pip install insightface onnxruntime opencv-python
```

`curl` must be available for the downloads.

## What to download

All data goes to `datasets/chokepoint/` at the repository root. This folder is in
`.gitignore`: do not commit data or prediction files.

| File | Size | Needed for | Time |
|---|---|---|---|
| `groundtruth.tar.xz` | 0.5 MB | the notebook (downloaded by the notebook) | a few seconds |
| `predictions/P2E_S1_C3.csv` | 0.4 MB | the notebook | from INF-49, or make it below |
| `P1E_S1.tar.xz` | 385 MB | making predictions only | depends on your connection |
| `P2E_S1.tar.xz` | 572 MB | making predictions only | depends on your connection |

Making the predictions took 3 min 49 s for the gallery and 7 min 12 s for
`P2E_S1_C3` (both days, 2,926 frames) on an Apple M1 Pro CPU.

## How to run

1. Put the prediction CSV at `datasets/chokepoint/predictions/P2E_S1_C3.csv`,
   or make it from the repository root:

   ```bash
   python -c "import sys; sys.path.insert(0, 'notebooks/chokepoint'); \
   from chokepoint_data import download_chokepoint; \
   download_chokepoint(['groundtruth.tar.xz', 'P1E_S1.tar.xz', 'P2E_S1.tar.xz'])"
   python notebooks/chokepoint/predict_insightface.py
   ```

2. Open and run the notebook from top to bottom:

   ```bash
   jupyter notebook notebooks/chokepoint/chokepoint_attendance.ipynb
   ```

The notebook downloads `groundtruth.tar.xz` itself. A second run skips the
download and extraction.
