
# Your model is not as good as you think. Or maybe it is. Without uncertainty, you cannot tell.

<img src="images/evaluation_metrics.jpg" alt="Machine Learning model accuracy" width="500"/>

Machine learning teams love clean numbers: **Accuracy of 92%**, **RMSE of 3.4**, **F1-score of 0.88**. These values look decisive, but they hide a critical truth: every metric is only an estimate. It depends on data sampling, class imbalance, noise, and evaluation protocol. **Confidence intervals** turn these fragile point estimates into something you can trust.

In this post, we will explore why confidence intervals matter in modern ML and DL, then introduce **Infer**, a project built to make metric uncertainty first-class. We will walk through how Infer works and how it supports both classification and regression metrics using a clean, end-to-end evaluation flow.

## Why confidence intervals matter in ML and DL

<img src="images/confidence_interval.png" alt="Confidence Interval" width="500"/>

A single metric value answers only one question: *what happened on this test set?*

Confidence intervals answer a more important one: *how reliable is this result?*

In real-world ML systems, uncertainty shows up everywhere:

- Small or imbalanced datasets inflate variance
- Cross-validation folds produce inconsistent scores
- Deployment data drifts from offline evaluation
- Two models with the same accuracy may behave very differently in production

Confidence intervals quantify this uncertainty by estimating a range within which the true metric value is likely to lie. Instead of saying “accuracy is 92%”, you say “accuracy is 92% ± 1.5% at 95% confidence”. This changes how decisions are made.

**Confidence intervals help you:**

- Compare models statistically, not emotionally
- Avoid overconfidence on noisy or small datasets
- Detect when improvements are not statistically meaningful
- Communicate reliability to stakeholders and regulators
- Make safer decisions in high-risk domains like healthcare or finance

Despite all this, CI computation for ML metrics is still fragmented, inconsistent, and often ignored.

> That gap is exactly why Infer exists.

## Introducing Infer

**Infer** is a Python-based evaluation framework designed to compute confidence intervals for machine learning metrics in a standardized, reliable, and extensible way.

Infer focuses on a simple idea: **metrics without uncertainty are incomplete**. The project provides a unified pipeline that starts from exploratory data analysis and ends with interpretable outputs that include metrics, confidence intervals, and visualizations.

At its core, Infer supports:

- Classification metrics like **Accuracy, Precision, Recall, F1-score, ROC-AUC**
- Regression metrics like **MAE, RMSE, MAPE**
- Multiple CI estimation methods including analytical and bootstrap-based approaches
- Automatic method selection based on metric and data characteristics
- Visual outputs that make uncertainty visible, not abstract

This design is rooted in the observation that CI computation has not been standardized across ML tooling, even though robust statistical methods are well-known and widely studied.

## The Infer evaluation flow

Infer follows a clean and intuitive evaluation pipeline, illustrated by the workflow diagram you provided:

**EDA → Evaluation Metric → Confidence Interval → Output**

Let’s break this down.

### 1. Exploratory Data Analysis (EDA)

Before computing any metric, Infer encourages understanding the data:

- Class balance for classification
- Distribution shape for regression targets
- Sample size and variance
- Potential sources of instability

EDA is critical because CI method choice depends heavily on data properties. For example, small sample sizes or skewed distributions often invalidate simple Normal approximations.

### 2. Metric computation

Infer computes standard ML metrics using well-defined formulations. It supports a broad range of metrics across ML domains, including regression and classification metrics commonly used in practice. At this stage, Infer produces the familiar point estimate. This is necessary but not sufficient.

### 3. Confidence interval estimation

This is where Infer differentiates itself.

Infer integrates multiple CI estimation methods, including:

- Analytical methods like Wald and Wilson intervals for proportion-based metrics
- Exact and Bayesian-inspired intervals for small samples
- Bootstrap-based methods, including bias-corrected and accelerated bootstrap, for complex metrics and regression errors
- Specialized methods for metrics like ROC-AUC and cross-validation settings

Each method has different assumptions, strengths, and failure modes. Infer documents these clearly and selects appropriate methods based on the metric and data characteristics.

### 4. Output and visualization

Infer does not stop at numbers. It generates:

- Metric values with confidence intervals
- Distribution plots from bootstrap samples
- CI bars and summaries for comparison
- Structured outputs suitable for reports and dashboards

This makes uncertainty visible and actionable, not buried in logs.

## Using Infer for classification metrics
<img src="images/classification.png" alt="Classification" width="400"/>

Classification metrics are often proportions or composite statistics. Infer supports confidence intervals for metrics such as:

- Accuracy
- Precision and Recall
- F1-score
- ROC-AUC

For example, in a binary classification experiment using Logistic Regression and XGBoost on the Breast Cancer Wisconsin dataset, Infer computed bootstrap confidence intervals for accuracy, precision, recall, and F1-score. The framework then validated whether true performance on a held-out validation set fell within the estimated CI ranges, achieving full coverage across all evaluated metrics.

This kind of validation is essential. A CI that looks narrow but fails to cover the true value is misleading. Infer explicitly measures coverage quality, not just interval width.

## Using Infer for regression metrics
<img src="images/linear_regression.png" alt="Regression" width="500"/>

Regression metrics often have complex, non-Normal distributions. Analytical CIs are usually unreliable here, which is why Infer emphasizes bootstrap methods for regression.

In one real-world case study involving IV-drip flow rate prediction, Infer evaluated model performance using **Mean Absolute Percentage Error (MAPE)**. The framework aggregated predictions and computed a bootstrap confidence interval around the MAPE estimate.

**The result:**

- MAPE of 7.88%
- 95% confidence interval of [6.47%, 9.84%]
- Corresponding accuracy interpretation of 92.12% with a well-defined uncertainty range

This gives far more insight than a single error number. It tells you not just how well the model performed, but how stable that performance is under resampling and noise.

## Why Infer matters

Infer is not just another metrics library. It represents a shift in how model evaluation is treated.

Instead of asking:

> “What is the metric?”

Infer encourages you to ask:

> “How confident are we in this metric?”

By combining rigorous statistical methods, practical ML metrics, and clear visual outputs, Infer helps teams:

- Make statistically defensible comparisons
- Avoid false improvements
- Communicate uncertainty clearly
- Build trust in ML systems

In a world where ML models increasingly inform critical decisions, **uncertainty is not optional**. Infer makes it measurable, visible, and usable.

