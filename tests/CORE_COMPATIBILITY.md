# INF-29 core API compatibility suite

Agreed with the user on 2026-11-03: start with Python **3.13.2**, classification and regression core APIs. This is the Team 1 starting scope, not evidence of a new bilateral agreement with Team 2.

Create a Python 3.13.2 virtual environment, activate it, and run from the repository root:

```sh
python --version
python -m pip install -r requirements/requirements.compatibility.txt
MPLBACKEND=Agg PYTHONPATH=src python -m pytest -q tests/test_core_api_compatibility.py
```

The suite checks list/NumPy inputs, the Wilson formula and confidence levels, classification scores and F1 averages against scikit-learn, regression scores against scikit-learn, deterministic bootstrap resampling, scalar return values, and public option validation.

Synthetic in-memory data are used. The suite does not download datasets or models and does not run object detection. Passing these API checks does not establish empirical confidence-interval coverage, compatibility of every dependency/OS, or success of the full existing suite. Existing tests and migration changes from PR #35 remain separate.

## Recorded validation

On 2026-11-03, CPython 3.13.2 on macOS arm64 against source commit `8dcaf9a`: **21 passed, 14 warnings in 47.67 seconds**. The warnings were Matplotlib/pyparsing deprecations. This was the targeted suite only, not the full existing test suite, and used source via PYTHONPATH rather than a new package installation. The core-only dependency subset above was installed in an isolated environment.
