# Confidence Intervals for evaluation metrics
![logo](logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a standardized evaluation tool that computes common machine learning metrics like Accuracy, MSE, and returns their confidence intervals using bootstraping and other methods. 

## The motivation

 Computing Confidence Intervals when providing performance estimates has not been sufficiently studied. Over the years, researchers have proposed multiple CI estimation methods in different fields, including medical statistics, AutoML, and text classification. Each method has specific assumptions, strengths, and limitations. However, most approaches are designed for a single metric or a narrow class of models, which limits their applicability. 
 
 We have developed a standardized evaluation tool that computes model performance metrics with their confidence intervals (CIs), including multiple CI calculation methods (bootstrapping, Wald, Wilson, etc.), and generating clear visualization graphs. It supports ML, CV, LLMs. 


## Installation

Clone this repository and install in development mode:

```bash
git clone https://github.com/humblebeeai/infer-ci.git
cd infer-ci
pip install -e .
```

### For YOLO Detection Metrics (Optional)

To use YOLO detection metrics with confidence intervals, install with the detection extras:

```bash
pip install -e .[detection]
```

This will install the additional dependencies (PyTorch, OpenCV, WandB, etc.) required for the customized Ultralytics YOLO CI validation.

## Getting started

```python
# All the possible imports:
from confidenceinterval import MetricEvaluator 
from confidenceinterval import accuracy_score, precision_score, tpr_score, f1_score
from confidenceinterval import mae, mse, rmse, r2_score, iou

# Classification example:

evaluate = MetricEvaluator()

accuracy, ci = evaluator.evaluate(
    y_true=y_true, 
    y_pred=y_pred, 
    task='classification', 
    metric='accuracy',
    method='wilson' ,
    # plot = True  Only supported in Bootstrap methods
)
```

## All methods can do an analytical computation, but do a bootsrapping computation by default
By default all the methods return an Bootstrap Bias-Corrected and Accelerated computation of the confidence intervals (CI). It is an advanced version of the basic bootstrap method that corrects for both bias and skewness in the estimate.

For other computation of the CI for any of the methods below, just specify method='jackknife_ci', or method='bootstrap_percentile'. These are different ways of doing the bootstrapping, but method='bootstrap_bca' is the generalibly reccomended method.

## Supported Methods
- **Analytical Methods**: Wilson, Normal, Agresti-Coull, Beta, Jeffreys, Binomial test
- **Bootstrap Methods**: Bootstrap Basic, Bootstrap Percentile, Bootstrap Bias-Corrected and Accelerated (Default)
- **Jackknife Method**: Jackknife

**Core Metrics Available via MetricEvaluator:**

**Classification:**
- `accuracy`: Overall classification accuracy
- `precision`: Positive Predictive Value (PPV) 
- `recall`: True Positive Rate/Sensitivity (TPR)
- `specificity`: True Negative Rate (TNR)
- `npv`: Negative Predictive Value
- `fpr`: False Positive Rate
- `f1`: F1 Score (supports averaging)
- `precision_takahashi`: Precision using Takahashi method
- `recall_takahashi`: Recall using Takahashi method
- `auc`: ROC AUC Score

**Regression:**
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `r2`: Coefficient of Determination
- `mape`: Mean Absolute Percentage Error
- `iou`: Intersection Over Union

**Detection (YOLO):**
- `yolo_map`: mAP50-95 per class with confidence intervals (bootstrap validation)
- `yolo_precision`: Precision per class with confidence intervals
- `yolo_recall`: Recall per class with confidence intervals 

You can also get available methods and metrics programmatically

```python
# Get available methods
classification_methods = evaluator.get_available_methods('classification')
regression_methods = evaluator.get_available_methods('regression')

print("Classification methods:", classification_methods)
print("Regression methods:", regression_methods)

# Get available metrics
classification_metrics = evaluator.get_available_metrics('classification')
regression_metrics = evaluator.get_available_metrics('regression')

print("Classification metrics:", classification_metrics)
print("Regression metrics:", regression_metrics)

```

The analytical computation here is using the (amazing) 2022 paper of Takahashi et al (reference below).
The paper derived recall and precision only for micro averaging.
We derive the recall and precision confidence intervals for macro F1 as well using the delta method.




## Detection Metrics (YOLO)

For object detection tasks using YOLO models, this library provides specialized metrics that compute confidence intervals through bootstrap validation. Instead of traditional y_true/y_pred arrays, these metrics take a YOLO model and dataset configuration.

### YOLO mAP with Confidence Intervals

```python
from confidenceinterval import MetricEvaluator
import numpy as np

evaluator = MetricEvaluator()

# Evaluate YOLO model - returns list of dicts (bootstrap samples)
stats_list = evaluator.evaluate(
    task='detection',
    metric='yolo_map',
    model='yolov8n.pt',   # Path to YOLO model weights
    data='coco128.yaml',  # Path to dataset YAML config
    n_iterations=100,     # Number of bootstrap iterations
    batch=32,
    imgsz=640
)

# Process stats_list to compute mean and CI for each class
print(f"\nYOLO mAP50-95 Results ({len(stats_list)} iterations):")
for class_name in stats_list[0].keys():
    # Collect values across iterations
    values = [stats[class_name] for stats in stats_list]
    mean_map = np.mean(values)
    ci = np.percentile(values, [2.5, 97.5])
    print(f"{class_name}: {mean_map:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

You can also use the metric functions directly:

```python
from confidenceinterval import yolo_map_with_ci
import numpy as np

# Direct function call - returns list of dicts
stats_list = yolo_map_with_ci(
    model='yolov8n.pt',
    data='coco128.yaml',
    n_iterations=100,
    confidence_level=0.95,
    wb_run_name='my_experiment',       # Optional: WandB logging
    project_name='yolo_experiments',   # Optional: WandB project
    project_entity='my_team'           # Optional: WandB entity
)

# prints results automatically
```

### How YOLO CI Validation Works

The YOLO confidence interval validation:
1. Runs validation `n_iterations` times (default: 100)
2. In each iteration, randomly samples ~50% of the validation data
3. Computes metrics (mAP/precision/recall) for each class in each iteration
4. Returns raw bootstrap samples as a list of dictionaries (one dict per iteration)
5. Automatically generates visualization plots and saves them to `runs/val/`
6. Optionally logs results to Weights & Biases

The DetectionCIValidator automatically:
- Saves all results to `{cwd}/runs/val/` directory
- Creates confidence interval plots (bar charts with error bars)
- Generates distribution histograms for each class
- Saves raw metrics to `ci_metrics.json`

You can then process the returned `stats_list` to compute mean and confidence intervals for any metric or class of interest.

## Get a Classification Report
The [classification_report.py](confidenceinterval%2Fclassification_report.py) function builds a text report showing the main classification metrics and their confidence intervals.
Each class will be first treated as a binary classification problem, the default CI for P and R used being Wilson, and Takahashi-binary for F1. Then the 
micro and macro multi-class metric will be calculated using the Takahashi-methods.

```python
from confidenceinterval import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)

     Class  Precision  Recall  F1-Score    Precision CI       Recall CI     F1-Score CI  Support
0  Class 0      0.600   1.000     0.750  (0.231, 0.882)    (0.439, 1.0)  (0.408, 1.092)        3
1  Class 1      0.889   1.000     0.941   (0.565, 0.98)    (0.676, 1.0)  (0.796, 1.086)        8
2  Class 2      1.000   0.667     0.800     (0.61, 1.0)  (0.354, 0.879)  (0.562, 1.038)        9
3    micro      0.850   0.850     0.850  (0.694, 1.006)  (0.694, 1.006)  (0.694, 1.006)       20
4    macro      0.830   0.889     0.830  (0.702, 0.958)  (0.775, 1.002)  (0.548, 1.113)       20
```

## Support for binary, macro and micro averaging for F1, Precision and Recall.
```python
from confidenceinterval import precision_score, recall_score, f1_score
binary_f1, ci = f1_score(y_true, y_pred, confidence_interval=0.95, average='binary')
macro_f1, ci = f1_score(y_true, y_pred, confidence_interval=0.95, average='macro')
micro_f1, ci = f1_score(y_true, y_pred, confidence_interval=0.95, average='micro')
bootstrap_binary_f1, ci = f1_score(y_true, y_pred, confidence_interval=0.95, average='binary', method='bootstrap_bca', n_resamples=5000)

```


----------

Citation
If you use this for research, please cite. Here is an example BibTeX entry:

```
@misc{infer-ci,
  title={Confidence intervals for evaluation metrics},
  author={Humblebee ai},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/humblebeeai/infer-ci}},
}
```

----------

## References
A python library for confidence intervals:
https://github.com/jacobgil/confidenceinterval

Confidence Interval Estimation of Predictive Performance in the Context of AutoML:
https://arxiv.org/abs/2406.08099v1

The binomial confidence interval computation uses the statsmodels package:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html

Yandex data school implementation of the fast delong method:
https://github.com/yandexdataschool/roc_comparison

X. Sun and W. Xu, "Fast Implementation of DeLong’s Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves," in IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389-1393, Nov. 2014, doi: 10.1109/LSP.2014.2337313:
https://ieeexplore.ieee.org/document/6851192

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP2

Confidence interval for micro-averaged F1 and macro-averaged F1 scores
`Kanae Takahashi,1,2 Kouji Yamamoto,3 Aya Kuchiba,4,5 and Tatsuki Koyama6`

B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, Boca Raton, FL, USA (1993)

http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
`Nathaniel E. Helwig, “Bootstrap Confidence Intervals”`


Bootstrapping (statistics), Wikipedia, https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
