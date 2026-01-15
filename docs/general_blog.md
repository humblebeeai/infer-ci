
# Your model is not as good as you think or maybe it is. Without uncertainty, you cannot tell.

Machine learning teams love clean numbers: **accuracy of 92 percent**. **RMSE of 3.4**, **F1 score of 0.88**. These numbers feel objective, precise, and reassuring. They fit neatly into dashboards, slide decks, and go or no-go decisions.

<p align="center"><img src="images/evaluation_metrics.jpg" alt="Machine Learning model accuracy" width="800" height="400" /></p>

But those values are not facts about the world. They are estimates derived from finite data, noisy labels, and a long chain of design choices such as sampling strategy, preprocessing, thresholds, and metric definitions. Treating a single number as definitive creates a **false sense of certainty**. That overconfidence often leads to poor decisions, unreliable deployments, missed regressions in minority groups, and wasted effort chasing noise during model selection.

Before we talk about confidence intervals, we need to step back and talk about evaluation itself. Why do we evaluate models, what are metrics actually telling us, and where do they quietly fail.

## Why we evaluate machine learning models at all

Model evaluation exists to inform decisions about whether and how a machine learning system should be used. Training a model produces parameters that optimize a loss function on observed data, but evaluation is the step that connects that optimization process to real world deployment, where mistakes have costs and performance expectations must be met.

In practice, evaluation helps teams determine whether a model meets the requirements of its intended use. It provides evidence about how the model behaves on data it was not trained on, how well it aligns with domain specific goals, and whether it satisfies constraints related to reliability, safety, or regulation. Without evaluation, there is no principled way to decide whether a model is ready to be deployed or how it compares to alternative approaches.

Evaluation also serves as a shared reference point across teams. It allows engineers, researchers, product managers, and stakeholders to reason about model behavior using a common framework, even if their priorities differ. In this sense, evaluation is not only a technical step but also an organizational tool that supports accountability and informed decision making throughout the lifecycle of a machine learning system.

## Metrics are abstractions, not truths

Evaluation metrics are simplified summaries of model behavior. Each metric captures one narrow notion of quality and ignores everything else.

Consider accuracy in a customer churn model. If only 5 percent of customers churn, a model that predicts “no churn” for everyone achieves 95 percent accuracy while providing zero business value. The metric looks strong precisely because it ignores the asymmetric cost of errors and the rarity of the outcome the model is meant to detect.

The same pattern appears across metrics. Accuracy assumes all errors have equal cost and that class distributions are stable. Precision and recall focus on different failure modes but ignore calibration and confidence. F1 score compresses competing objectives into a single value while hiding tradeoffs. Regression metrics like RMSE and MAE emphasize different error magnitudes and implicitly encode assumptions about loss functions.

Choosing the right metric already requires domain understanding, cost modeling, and clarity about how predictions will be used. A fraud detection system, a medical diagnostic model, and a recommender system should not be evaluated the same way even if they use similar algorithms.

## The hidden assumption behind point metrics

When metric values are reported, they are often treated as fixed properties of a model, similar to architectural details such as the number of parameters or layers. Saying that a model has 92 percent accuracy or an F1 score of 0.88 implicitly suggests that these numbers are stable characteristics that can be relied on as precise summaries of performance.

In reality, every reported metric is an estimate that depends on a specific evaluation setup. The value reflects the particular test sample that was drawn, the labels that were collected and possibly contain noise, and the operational choices embedded in the evaluation process, including preprocessing steps, thresholds, and metric definitions. None of these factors are fixed in practice, and small changes in any of them can alter the resulting metric.

If the same model were evaluated on a slightly different test set drawn from the same underlying data distribution, the reported metric would almost certainly change. Sometimes the change would be minor, but in other cases it could be large enough to alter conclusions about model quality or readiness for deployment. Treating a single point estimate as definitive therefore relies on an implicit assumption that this variability is negligible, an assumption that often does not hold in real machine learning systems.

## Where single metric evaluation fails in practice

The limitations of point metrics show up repeatedly across real world machine learning systems.

#### Small or medium sized test sets

When test sets are small, randomness dominates. On a test set of a few hundred samples, one or two additional errors can materially change reported performance. Teams routinely interpret these fluctuations as meaningful improvements or regressions when they are simply noise.

#### Imbalanced problems

In domains like fraud detection, medical diagnosis, or anomaly detection, the most important class is often the rare one. Metrics such as recall, precision, and F1 score can swing dramatically with small changes in class composition or labeling errors. A single point estimate can easily mask extreme instability in minority class performance.

#### Threshold sensitivity

Many metrics depend on a chosen decision threshold. A model that looks strong at one threshold may look weak at another. Reporting a single number hides how sensitive the model is to operational choices that will inevitably change over time.

#### Model selection and hyperparameter tuning

When dozens or hundreds of models are tried, the best looking metric is often the luckiest one. Selecting based on point estimates alone systematically favors noise. This is one of the most common sources of overestimated offline performance.

#### Subgroup and fairness analysis

Aggregate metrics can look stable while subgroup performance is wildly uncertain. Small sample sizes within demographic slices amplify variance. Without explicitly accounting for uncertainty, teams often miss fragile or unsafe behavior in exactly the groups that matter most.

## The real question metrics do not answer

A single metric value answers a narrow and retrospective question: *what did the model score on this particular evaluation sample?* 
That information is useful, but it is rarely sufficient for making real decisions.

Most practical decisions depend on a forward looking question instead. *How much could this metric plausibly change if the data were slightly different, if labels contained noise, or if operating conditions shifted in production?* Without an answer to that question, teams cannot reliably judge whether an observed improvement is meaningful, whether a regression reflects real degradation, or whether a model is safe to deploy.

This gap between reporting a number and understanding its stability is where many evaluation failures originate. Metrics summarize past performance, but decisions require an understanding of uncertainty. Confidence intervals exist precisely to bridge this gap, by turning point estimates into ranges that reflect how much trust we can place in them.

## From point estimates to uncertainty aware evaluation

**Confidence intervals** transform evaluation from a static report into a probabilistic statement.

Instead of saying the model has 92 percent accuracy, we say that the accuracy is likely to lie within a certain range given the data we observed. That range captures sampling variability, measurement noise, and sensitivity to evaluation choices.

This shift changes how metrics are used. Metrics stop being treated as precise truths and start being treated as uncertain estimates. That single change has far reaching consequences for deployment decisions, monitoring, and model comparison.

## Why confidence intervals matter in machine learning

A single metric value summarizes what a model achieved on a specific evaluation sample, but it does not indicate how stable or reliable that result is. In practice, most decisions depend not only on the reported number but also on how much that number could change under slightly different conditions.

Confidence intervals address this by quantifying the range within which a metric is likely to fall, given the inherent variability of finite data, noisy labels, and evaluation design choices. This is not a purely statistical concern. The width and position of an interval directly affect real decisions about deployment, monitoring, and prioritization.

By exposing best case and worst case scenarios, confidence intervals allow teams to translate abstract metrics into concrete business risk. They reduce the chance of selecting models that appear better only due to noise during hyperparameter tuning. They make it possible to distinguish expected metric fluctuation from genuine performance degradation after deployment. They also highlight instability in subgroups that aggregate metrics can hide, improving fairness analysis and regulatory confidence. Finally, confidence intervals improve communication by replacing overconfident point estimates with calibrated statements about what the data actually supports.

## Confidence intervals in real decision making

Consider a healthcare model that reports 94 percent sensitivity on a test set. On its own, that number sounds strong.

With a confidence interval, the picture changes. The sensitivity might plausibly lie between 87 percent and 98 percent. That lower bound may be unacceptable in a clinical setting where missed diagnoses carry serious consequences. Without the interval, that risk remains invisible.

This logic is already standard in online experimentation. No serious A/B testing team would ship a change based on point estimates alone. Offline model evaluation is answering the same question: *is this difference real, or is it noise?*

Confidence intervals bring the same discipline to machine learning evaluation.

## Where naive confidence interval approaches fail

Not all confidence intervals (CI) are created equal. Many common approaches break down in realistic ML settings.

Metrics like accuracy and recall are not normally distributed in small samples. Bootstrapping can fail when data are highly imbalanced or correlated. Some metrics are discontinuous with respect to thresholds. Others depend on complex pipelines that violate independence assumptions.

This is why confidence intervals need to be metric aware, data aware, and production ready. Treating CI computation as an afterthought often produces misleading results that are worse than no interval at all.

## Introducing Infer

<p align="center"><img src="images/infer_black.png" alt="infer" width="1000" height="300" /></p>

**Infer** is a Python-based evaluation framework that turns point metrics into **uncertainty-aware performance statements**. Instead of returning a single score, Infer computes statistically appropriate confidence intervals, making model evaluation and comparison more reliable in real-world settings. 

### What makes Infer different

Infer is built on a simple insight: **metrics without uncertainty quantification create statistical and business risk**.
Naive bootstrapping or default statistical methods often break silently when datasets are small, imbalanced, or when metrics violate their assumptions. Infer abstracts this complexity away.

* **Automatic CI-method selection**:
  Infer selects confidence-interval methods that match each metric’s statistical properties, handling edge cases such as small samples, class imbalance, and zero-division safely.

* **Metric-aware statistical assumptions**:
  Different metrics require different uncertainty models. Infer encodes these distinctions so practitioners do not need to reason about statistical validity metric by metric.

* **Consistent, unified APIs**:
  The same interface works across classification, regression, and detection tasks, reducing evaluation friction and error-prone custom code.

* **Reproducible evaluation outputs**:
  Structured results are designed for reports, dashboards, and stakeholder communication, not just ad-hoc experimentation.


### Installation
#### From PyPI (Recommended)
```Python
pip install infer-ci
```

### From Source (Development)
```Python
git clone https://github.com/humblebeeai/infer-ci.git
cd infer-ci
pip install -e ".[dev]"
```

### Getting started
```Python
# All the possible imports:
from infer_ci import MetricEvaluator
from infer_ci import accuracy_score, precision_score, tpr_score, f1_score
from infer_ci import mae, mse, rmse, r2_score, iou

# Classification example:
evaluator = MetricEvaluator()

accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
    # plot=True  # Supported in Bootstrap methods
)
```

**The result:**

- MAPE of 7.88%
- 95% confidence interval of [6.47%, 9.84%]
- Corresponding accuracy interpretation of 92.12% with a well-defined uncertainty range

This gives far more insight than a single error number. It tells you not just how well the model performed, but how stable that performance is under resampling and noise.

### Conclusion

Machine learning evaluation is ultimately about decision making under uncertainty, yet most evaluation practices pretend that uncertainty does not exist. Point metrics give the illusion of precision while hiding the variability that determines whether a model improvement is real, reproducible, and safe to deploy. Confidence intervals restore that missing context by turning metrics into statements about reliability, risk, and stability rather than isolated numbers. Infer operationalizes this shift, making uncertainty-aware evaluation practical, consistent, and statistically sound across real-world ML workflows. In an era where models increasingly influence high-stakes decisions, treating uncertainty as optional is no longer defensible and Infer makes it unavoidable, measurable, and actionable.
