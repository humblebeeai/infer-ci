# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Migrated all imports from `confidenceinterval` to `infer_ci`
- **BREAKING**: Package now only supports `infer_ci` import path
- Renamed test files to follow pytest conventions (`test_*.py`)
- Updated all examples to use `infer_ci` imports

### Removed
- Removed cookiecutter template boilerplate files (`_base.py`, `config.py`, `_utils.py`, `__main__.py`)
- Removed legacy `confidenceinterval/` directory
- Removed empty `src/modules/` directory
- Removed root-level `__init__.py` with broken imports
- Removed template documentation (`MyClass.md`)

### Fixed
- Fixed import errors in README.md examples
- Fixed variable name consistency in documentation
- Fixed broken imports in 8 test files
- Added missing Python version and PyPI badges to README

### Added
- Created comprehensive simple classification example (`examples/simple/main.py`)
- Created advanced regression example with multiple metrics (`examples/advanced/regression_example.py`)
- Added PyPI installation instructions to README
- Added development installation instructions to README

## [0.2.0] - 2024-12-20

### Added
- **Object Detection Metrics**: Added support for YOLO detection evaluation metrics ([#6](https://github.com/humblebeeai/infer-ci/pull/6))
  - `map` (mean Average Precision @0.5:0.95)
  - `map50` (mean Average Precision @0.5)
  - Detection precision and recall metrics
  - Per-class confidence interval calculation support
- Comprehensive object detection metrics usage guide (`docs/object-detection-metrics-usage.md`)
- Integration with Ultralytics YOLO for model evaluation
- Bootstrap confidence intervals for detection metrics

### Changed
- Refactored metric names from `yolo_map` to `map` for clarity
- Enhanced YOLO detection metrics with confidence intervals
- Improved AP (Average Precision) calculation accuracy

### Fixed
- Fixed YOLO detection metrics AP calculation integration

## [0.1.5] - 2024-10-15

### Added
- Classification report with confidence intervals (`classification_report_with_ci`)
- Pandas integration for classification reports
- Comprehensive documentation for classification reports with examples

### Fixed
- Fixed numpy negation syntax in `compute_ground_truth_statistics` ([#10](https://github.com/humblebeeai/infer-ci/pull/10))
- Improved pandas operations to avoid FutureWarning (replaced concat with more efficient approach)

### Changed
- Relaxed package requirements for better compatibility

## [0.1.4] - 2024-09-01

### Added
- Visualization function for bootstrap methods
- IoU (Intersection over Union) metric for regression tasks
- Developer guide for adding new metrics

### Changed
- Updated project metadata
- Renamed results directory for consistency
- Updated documentation structure (`Docs` ’ `docs`)

### Fixed
- Fixed relative import errors in test files

## [0.1.3] - 2024-08-15

### Added
- Comprehensive documentation files:
  - `Methods.md` - CI calculation methods
  - `Metrics.md` - Available metrics
  - `Schemes.md` - Evaluation schemes
- Example datasets for testing (IV-Drip dataset)
- Example usage files demonstrating real-world applications

### Fixed
- Fixed 5000+ lines of documentation rendering/formatting errors
- Fixed scheme documentation errors

## [0.1.2] - 2024-07-20

### Added
- GitHub Actions workflow for automated testing and linting
- Python 3.10+ support verification

### Changed
- Updated Python version requirements in CI/CD (3.10.0 ’ 3.11)

## [0.1.1] - 2024-06-15

### Added
- Regression metrics with confidence intervals:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Mean Absolute Percentage Error (MAPE)
- Example files demonstrating regression metrics usage
- Test datasets for regression evaluation

### Changed
- Updated `.gitignore` to include virtual environment directories
- Added xgboost dependency to requirements

## [0.1.0] - 2024-05-01

### Added
- Initial release of confidence intervals library for Python
- Core `MetricEvaluator` interface for unified metric evaluation
- **Classification Metrics**:
  - Accuracy with Wilson, Normal, Agresti-Coull, Beta, Jeffreys methods
  - Precision (PPV) with confidence intervals
  - Recall (TPR/Sensitivity) with confidence intervals
  - Specificity (TNR) with confidence intervals
  - Negative Predictive Value (NPV)
  - False Positive Rate (FPR)
  - F1 Score with support for binary, macro, and micro averaging
  - ROC AUC Score with DeLong method
  - Takahashi methods for precision and recall
- **Bootstrap Methods**:
  - Bootstrap Basic
  - Bootstrap Percentile
  - Bootstrap Bias-Corrected and Accelerated (BCa) - Default method
- **Jackknife Method** for CI estimation
- **Analytical Methods** for binomial proportions
- Support for binary, macro, and micro averaging for F1, Precision, Recall
- Fast DeLong implementation for AUC comparison
- MIT License
- Comprehensive README with examples
- Initial test suite

### Dependencies
- Python 3.10+
- NumPy, SciPy, Pandas
- scikit-learn
- statsmodels
- matplotlib (for visualizations)

---

## Migration Guide (0.1.x ’ 0.2.x)

### Import Changes

If you're upgrading from an earlier version that used `confidenceinterval`:

```python
# OLD (0.1.x and earlier)
from confidenceinterval import MetricEvaluator
from confidenceinterval import accuracy_score, precision_score

# NEW (0.2.x and later)
from infer_ci import MetricEvaluator
from infer_ci import accuracy_score, precision_score
```

All functionality remains the same - only the import path has changed.

---

## Links

- [PyPI Package](https://pypi.org/project/infer-ci/)
- [GitHub Repository](https://github.com/humblebeeai/infer-ci)
- [Documentation](https://infer.humblebee.ai)
- [Issue Tracker](https://github.com/humblebeeai/infer-ci/issues)

[Unreleased]: https://github.com/humblebeeai/infer-ci/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/humblebeeai/infer-ci/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/humblebeeai/infer-ci/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/humblebeeai/infer-ci/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/humblebeeai/infer-ci/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/humblebeeai/infer-ci/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/humblebeeai/infer-ci/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/humblebeeai/infer-ci/releases/tag/v0.1.0
