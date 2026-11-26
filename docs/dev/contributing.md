---
title: Contributing
---

# 🤝 Contributing

Thank you for your interest in contributing to infer-ci! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Adding New Metrics](#adding-new-metrics)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Testing Requirements](#testing-requirements)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/infer-ci.git
   cd infer-ci
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/humblebeeai/infer-ci.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package in editable mode with all development dependencies including:
   - pytest (testing framework)
   - pytest-cov (coverage reporting)
   - pyright (type checking)
   - pre-commit (runs black, flake8 and the other checks, see [Pre-commit Hooks](#pre-commit-hooks))
   - mkdocs and plugins (documentation)

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Running Tests

### Run all tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=infer_ci --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in your browser.

### Run specific test files

```bash
pytest tests/test_classification.py
pytest tests/test_regression.py
```

### Run tests with verbose output

```bash
pytest -v
```

### Using the test script

The project includes a convenience script for running tests:

```bash
./scripts/test.sh        # Run all tests
./scripts/test.sh -c     # Run with coverage
./scripts/test.sh -l     # Show log output
```

## Code Style

This project follows strict code quality standards:

### Formatting

We use **Black** for code formatting, with its default line length (88). It runs as a pre-commit hook:

```bash
pre-commit run black --all-files
```

### Linting

We use **flake8** for linting, with a maximum line length of 120. It runs as a pre-commit hook:

```bash
pre-commit run flake8 --all-files
```

### Type Checking

We use **pyright** for static type checking:

```bash
python -m pyright src/infer_ci
```

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit: Black, flake8, pyright, and checks such as large files (over 2.5 MB) and commits to `main`. If you installed them, they will run automatically. To run manually:

```bash
pre-commit run --all-files
```

## Adding New Metrics

Adding a new metric involves several steps. Here's a complete guide:

### 1. Determine the Metric Type

First, identify which category your metric belongs to:
- **Classification**: Metrics for classification tasks (accuracy, precision, recall, F1, etc.)
- **Regression**: Metrics for regression tasks (MAE, MSE, RMSE, R², etc.)
- **Detection**: Metrics for object detection (mAP, precision, recall)

### 2. Implement the Metric Function

Create your metric function in the appropriate module:

**For classification metrics**: Add to `src/infer_ci/binary_metrics.py` (proportion-based metrics) or `src/infer_ci/takahashi_methods.py` (precision, recall, F1)

**For regression metrics**: Add to `src/infer_ci/regression_metrics.py`

**Example template** (a regression metric, using the shared helper in `regression_metrics.py`):

```python
from typing import List, Tuple, Union

import numpy as np


def my_new_metric(y_true: List[float],
                  y_pred: List[float],
                  confidence_level: float = 0.95,
                  method: str = 'bootstrap_bca',
                  compute_ci: bool = True,
                  **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Compute my new metric and optionally the confidence interval.

    Parameters
    ----------
    y_true : List[float]
        The ground truth target values.
    y_pred : List[float]
        The predicted target values.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    compute_ci : bool, optional
        If True return the confidence interval as well as the metric value, by default True
    **kwargs
        Passed on to the bootstrap, for example n_resamples or random_state.

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric value, or (value, (lower, upper)) if compute_ci is True.

    Examples
    --------
    >>> y_true = [3.0, 5.0, 2.5, 7.0]
    >>> y_pred = [2.5, 5.0, 3.0, 8.0]
    >>> value, (lower, upper) = my_new_metric(y_true, y_pred, random_state=0)
    """
    def my_new_metric_func(y_true, y_pred):
        # Calculate your metric from the two arrays
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    return _compute_metric_with_optional_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=my_new_metric_func,
        confidence_level=confidence_level,
        method=method,
        compute_ci=compute_ci,
        **kwargs
    )
```

### 3. Register the Metric in MetricEvaluator

Import your metric at the top of `src/infer_ci/evaluator.py` and add it to the matching dictionary in `MetricEvaluator.__init__` (`classification_metrics`, `regression_metrics` or `detection_metrics`). Without this step, `evaluate_metric(..., metric='my_new_metric')` raises "Unknown metric".

```python
from .regression_metrics import (
    # ... existing imports ...
    my_new_metric,
)

# in MetricEvaluator.__init__
self.regression_metrics = {
    # ... existing metrics ...
    'my_new_metric': my_new_metric,
}
```

### 4. Export the Metric

Add your metric to `src/infer_ci/__init__.py`:

```python
from .regression_metrics import (
    # ... existing imports ...
    my_new_metric,
)

__all__ = [
    # ... existing exports ...
    'my_new_metric',
]
```

### 5. Write Tests

Create comprehensive tests in the appropriate test file:

```python
# tests/test_my_metric.py
import numpy as np
import pytest
from infer_ci import my_new_metric, evaluate_metric


def _data():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(1, 10, size=60)
    y_pred = y_true + rng.normal(0, 1, size=60)  # realistic data, with noise
    return y_true, y_pred


def test_my_new_metric_basic():
    """Test basic functionality."""
    y_true, y_pred = _data()
    value, (lower, upper) = my_new_metric(y_true, y_pred, random_state=0)
    assert lower <= value <= upper


def test_my_new_metric_unified_interface():
    """evaluate_metric gives the same result as the direct call."""
    y_true, y_pred = _data()
    unified = evaluate_metric(y_true, y_pred, task='regression', metric='my_new_metric',
                              compute_ci=False)
    assert unified == pytest.approx(my_new_metric(y_true, y_pred, compute_ci=False))


def test_my_new_metric_edge_cases():
    """Test edge cases."""
    # Perfect predictions
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert my_new_metric(y, y, compute_ci=False) == 0.0
```

### 6. Add Documentation

Create API documentation in `docs/api-docs/`:

```markdown
# my_new_metric

::: infer_ci.binary_metrics.my_new_metric
    options:
      show_root_heading: true
      show_source: true

## Examples

[Your examples here]

## Mathematical Definition

[Your formula here]

## References

[Your citations here]
```

### 7. Update CHANGELOG.md

Add an entry to the `[Unreleased]` section:

```markdown
### Added
- New metric: `my_new_metric` for [task type] with confidence intervals
```

## Documentation

### Building Documentation Locally

```bash
mkdocs serve
```

Then visit `http://127.0.0.1:8000` in your browser.

### Documentation Structure

- `docs/getting-started/` - Tutorials and quick starts
- `docs/api-docs/` - API reference documentation
- `docs/dev/` - Developer documentation
- `README.md` - Main package documentation

### Writing Documentation

- Use clear, concise language
- Include code examples
- Explain mathematical concepts when relevant
- Add references to papers/sources

## Pull Request Process

1. **Create a branch in your fork**, starting from the latest `dev`:
   ```bash
   git fetch upstream
   git checkout -b feature/my-new-feature upstream/dev
   ```
   Don't work on `main`, and don't create branches in the `humblebeeai` repository.

2. **Make your changes** following the code style guidelines

3. **Write or update tests** for your change. Every test needs at least one `assert`.

4. **Update documentation** as needed. If you change an English page that has a Korean version (`*.ko.md`), update both.

5. **Run tests and checks**:
   ```bash
   python -m pytest tests/
   pre-commit run --all-files
   ```

6. **Commit your changes** following the commit message guidelines

7. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

8. **Open a Pull Request** to **`humblebeeai/infer-ci`, base branch `dev`** (not `main`). Check both on the GitHub "Open a pull request" page before you create it. Fill in the PR template with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues
   - Screenshots/examples if applicable

9. **Address review feedback** if requested. If a reviewer pushed commits to your branch, run `git pull` before you push again.

10. **Wait for CI checks** to pass before merging. CI runs only part of the test suite, so always run `python -m pytest tests/` locally too.

### Before you open a PR

- One issue per PR. Put unrelated work, such as a new notebook, in its own PR.
- Describe only what the diff changes.
- Don't commit large files such as model weights (the pre-commit hook blocks files over 2.5 MB).
- Put notebooks in `notebooks/`, with no spaces in file names.
- Read AI-generated text and code before you commit it.
- Set `random_state` in examples and tests so results can be reproduced.

## Commit Message Guidelines

We follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(metrics): add IoU metric for regression tasks

Implemented Intersection over Union metric with bootstrap
confidence intervals for regression evaluation.

Closes #123
```

```
fix(bootstrap): correct BCa interval calculation

Fixed bias correction in bootstrap BCa method that was
causing incorrect confidence intervals for small samples.
```

```
docs(api): add examples for F1 score usage

Added comprehensive examples showing binary, macro, and
micro averaging for F1 score computation.
```

## Testing Requirements

### Coverage Standards

- **Minimum coverage**: 80% overall
- **New code**: Should have at least 90% coverage
- **Critical paths**: Must have 100% coverage

### What to Test

1. **Basic functionality**: Does it work as expected?
2. **Edge cases**: Empty arrays, perfect scores, all zeros
3. **Invalid inputs**: Wrong shapes, invalid parameters
4. **Integration**: Works through MetricEvaluator
5. **Reproducibility**: Random state produces consistent results
6. **CI methods**: Different methods produce valid intervals

### Test Organization

- `tests/test_core_api_compatibility.py` - Core API checks (Wilson, scikit-learn agreement, F1 averaging, errors)
- `tests/test_sklearn_equality.py` - Metric values compared with scikit-learn
- `tests/test_delong.py` - DeLong AUC method
- `tests/test_classification.py`, `tests/test_binary_classification_ci.py` - Classification pipelines
- `tests/test_regression.py`, `tests/test_regression_ci.py` - Regression pipelines
- `tests/test_object_detection_ci.py` - Detection with a YOLO model (needs `ultralytics` and `datasets/coco128`)
- `tests/conftest.py` - Shared settings (Matplotlib `Agg` backend)

## Questions?

If you have questions:

1. Check the [documentation](https://infer.humblebee.ai)
2. Search [existing issues](https://github.com/humblebeeai/infer-ci/issues)
3. Open a [new issue](https://github.com/humblebeeai/infer-ci/issues/new)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
