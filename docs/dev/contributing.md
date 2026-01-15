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
   - black (code formatting)
   - flake8 (linting)
   - mypy (type checking)
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
./scripts/test.sh -l     # Run linting checks
```

## Code Style

This project follows strict code quality standards:

### Formatting

We use **Black** for code formatting with a line length of 100:

```bash
black src/infer_ci tests
```

### Linting

We use **flake8** for linting:

```bash
flake8 src/infer_ci tests
```

### Type Checking

We use **mypy** for static type checking:

```bash
mypy src/infer_ci
```

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit. If you installed them, they will run automatically. To run manually:

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

**For classification metrics**: Add to `src/infer_ci/binary_metrics.py` or `src/infer_ci/multiclass_metrics.py`

**For regression metrics**: Add to `src/infer_ci/regression_metrics.py`

**Example template**:

```python
from typing import Tuple, Optional
import numpy as np
from .methods import compute_ci

def my_new_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'bootstrap_bca',
    confidence_interval: float = 0.95,
    n_resamples: int = 2000,
    compute_ci: bool = True,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute my new metric with confidence interval.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    method : str, default='bootstrap_bca'
        Method for computing confidence interval
    confidence_interval : float, default=0.95
        Confidence level (0 < CI < 1)
    n_resamples : int, default=2000
        Number of bootstrap resamples
    compute_ci : bool, default=True
        Whether to compute confidence interval
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    metric_value : float
        The computed metric value
    ci : Tuple[float, float]
        Confidence interval (lower, upper)

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 0])
    >>> value, ci = my_new_metric(y_true, y_pred)
    >>> print(f"Metric: {value:.3f}, CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """
    # Your metric computation logic here
    def metric_func(y_true, y_pred):
        # Calculate your metric
        return result

    metric_value = metric_func(y_true, y_pred)

    if not compute_ci:
        return metric_value, (np.nan, np.nan)

    ci = compute_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=metric_func,
        method=method,
        confidence_interval=confidence_interval,
        n_resamples=n_resamples,
        random_state=random_state,
        **kwargs
    )

    return metric_value, ci
```

### 3. Register the Metric in MetricEvaluator

Add your metric to `src/infer_ci/evaluator.py`:

```python
# In the appropriate task section (classification/regression/detection)
CLASSIFICATION_METRICS = {
    # ... existing metrics ...
    'my_new_metric': my_new_metric,
}
```

### 4. Export the Metric

Add your metric to `src/infer_ci/__init__.py`:

```python
from .binary_metrics import (
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
from infer_ci import my_new_metric, MetricEvaluator

def test_my_new_metric_basic():
    """Test basic functionality."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])

    value, ci = my_new_metric(y_true, y_pred)

    assert isinstance(value, float)
    assert len(ci) == 2
    assert ci[0] <= value <= ci[1]

def test_my_new_metric_evaluator():
    """Test through MetricEvaluator interface."""
    evaluator = MetricEvaluator()
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])

    value, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric='my_new_metric',
        method='wilson'
    )

    assert value > 0

def test_my_new_metric_edge_cases():
    """Test edge cases."""
    # Perfect predictions
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    value, _ = my_new_metric(y_true, y_pred, compute_ci=False)
    assert value == 1.0

    # All wrong
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    value, _ = my_new_metric(y_true, y_pred, compute_ci=False)
    assert value == 0.0
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

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the code style guidelines

3. **Write or update tests** to maintain coverage above 80%

4. **Update documentation** as needed

5. **Run tests and checks**:
   ```bash
   pytest --cov=infer_ci
   black src/infer_ci tests
   flake8 src/infer_ci tests
   ```

6. **Commit your changes** following the commit message guidelines

7. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

8. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues
   - Screenshots/examples if applicable

9. **Address review feedback** if requested

10. **Wait for CI checks** to pass before merging

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

- `tests/test_classification.py` - Classification metrics
- `tests/test_regression.py` - Regression metrics
- `tests/test_detection.py` - Detection metrics
- `tests/test_methods.py` - CI calculation methods
- `tests/test_evaluator.py` - MetricEvaluator interface

## Questions?

If you have questions:

1. Check the [documentation](https://infer.humblebee.ai)
2. Search [existing issues](https://github.com/humblebeeai/infer-ci/issues)
3. Open a [new issue](https://github.com/humblebeeai/infer-ci/issues/new)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
