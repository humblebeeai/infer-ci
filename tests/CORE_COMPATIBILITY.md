# Core API compatibility

Create a Python 3.13.2 virtual environment, activate it, and run from the repository root:

```sh
python --version
python -m pip install -r requirements/requirements.compatibility.txt
MPLBACKEND=Agg PYTHONPATH=src python -m pytest -q tests/test_core_api_compatibility.py
```

The suite checks list/NumPy inputs, the Wilson formula and confidence levels, classification scores and F1 averages against scikit-learn, regression scores against scikit-learn, deterministic bootstrap resampling, scalar return values, and public option validation.

Synthetic in-memory data are used. The suite does not download datasets or models and does not run object detection. Passing these API checks does not establish empirical confidence-interval coverage, compatibility of every dependency/OS, or success of the full existing suite. Existing tests and migration changes from PR #35 remain separate.

## Automated PR checks

`.github/workflows/core-python313.yml` runs this targeted suite on Ubuntu 24.04 / Python 3.13.2 for pull requests touching the core source, suite, dependency subset or workflow. It also supports main-branch pushes and manual dispatch. The job has read-only repository permissions and performs no publishing or deployment. A fork PR may require a maintainer to approve the initial workflow run; local results do not imply a passed GitHub run.
