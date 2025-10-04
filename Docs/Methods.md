# Documentation on Confidence Interval Evaluation Methods for ML Metrics

This documentation provides a structured overview of **confidence interval (CI) methods** used to evaluate machine learning metrics. It explains, step by step, how different statistical approaches quantify the uncertainty of performance metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and others. Starting from simple classical formulas and moving to more advanced and robust techniques, each method is presented with its theory, computation, advantages, limitations, pitfalls, and use cases in ML evaluation. The goal is to create a clear, practical reference for selecting the right CI method when building reliable ML evaluation tools.

## Table of Contents

- [Documentation on Confidence Interval Evaluation Methods for ML Metrics](#documentation-on-confidence-interval-evaluation-methods-for-ml-metrics)
  - [📋 Comprehensive CI Method Guide (Quick Summary Table)](#-comprehensive-ci-method-guide-for-ml-metrics)
  - [📊 Summary of CI Evaluation Methods (By Method)](#-summary-of-ci-evaluation-methods-for-ml-metrics)
  - [🔹 Advanced Extensions](#-advanced-extensions-beyond-the-13-core-methods)
  - [📚 References & Resources](#-references--resources)

- [Method 1 — Normal/Wald Confidence Interval](#method-1--normalwald-confidence-interval)
- [Method 2 — Wilson Score Confidence Interval](#method-2--wilson-score-confidence-interval)
- [Method 3 — Clopper–Pearson (Exact) Confidence Interval](#method-3--clopperpearson-exact-confidence-interval)
- [Method 4 — Agresti–Coull Confidence Interval](#method-4--agresticoull-confidence-interval)
- [Method 5 — Jeffreys Confidence Interval](#method-5--jeffreys-confidence-interval)
- [Method 6 — Likelihood-Ratio Confidence Interval](#method-6--likelihood-ratio-confidence-interval)
- [Method 7 — Bootstrap Percentile Confidence Interval](#method-7--bootstrap-percentile-confidence-interval)
- [Method 8 — Bootstrap BCa (Bias-Corrected and Accelerated) Confidence Interval](#method-8--bootstrap-bca-bias-corrected-and-accelerated-confidence-interval)
- [Method 9 — Bootstrap Normal Confidence Interval](#method-9--bootstrap-normal-confidence-interval)
- [Method 10 — Jackknife-Normal Confidence Interval](#method-10--jackknife-normal-confidence-interval)
- [Method 11 — DeLong’s Confidence Interval for ROC-AUC](#method-11--delongs-confidence-interval-for-roc-auc)
- [Method 12 — Nadeau–Bengio Correction (Cross-Validation Confidence Interval)](#method-12--nadeaubengio-correction-cross-validation-confidence-interval)
- [Method 13 — Bouckaert–Frank Correction (Repeated Cross-Validation Confidence Interval)](#method-13--bouckaertfrank-correction-repeated-cross-validation-confidence-interval)


# Method 1 — Normal/Wald Confidence Interval

## 1. What it is
The **Normal (Wald) confidence interval** is the simplest and oldest method:

- Assumes the sampling distribution of your metric is approximately **Normal** when sample size is large.
- Formula:

$$
\hat{\theta} \pm z_{1-\alpha/2} \cdot SE(\hat{\theta})
$$

Where:
- $\hat{\theta}$ = metric estimate (e.g., accuracy = 0.84)  
- $SE(\hat{\theta})$ = estimated standard error  
- $z_{1-\alpha/2}$ = critical Normal value (≈1.96 for 95% CI)

---

## 2. How to compute
Example: **Accuracy (binary classification)**

1. Compute accuracy:

$$
\hat{p} = \frac{k}{n}
$$

where $k$ = number of correct predictions, and $n$ = total samples.

2. Estimate standard error:

$$
SE(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

3. Form CI:

$$
CI = \hat{p} \pm z \cdot SE(\hat{p})
$$

Example:  
- Accuracy = 0.84 on $n = 500$ samples  
- $SE = \sqrt{0.84 \cdot 0.16 / 500} \approx 0.016$  
- CI = $0.84 \pm 1.96 \cdot 0.016 = [0.81, 0.87]$

---

## 3. When it works
- Large test sets ($n \geq 30$–50).  
- Metrics that are simple **proportions** (accuracy, precision, recall, specificity).  
- Quick baseline CI when a rough sense of uncertainty is enough.

---

## 4. Pros
- Very **easy** to compute.  
- **Fast** (no resampling needed).  
- Works decently for balanced datasets with large $n$.  
- Simple to explain in reports.

---

## 5. Cons
- Poor coverage when metric is near 0 or 1.  
- Breaks down for **small $n$** or **imbalanced data**.  
- Can produce impossible CIs (e.g., negative accuracy, >1 recall).  
- Too optimistic in many ML settings.

---

## 6. Pitfalls
- Do not use if $n$ is small (<30 per class).  
- Avoid for skewed/non-Normal metrics (F1, AUC, log-loss).  
- Underestimates uncertainty → misleading results.

---

## 7. Use in ML metrics
- Acceptable for **accuracy** on large test sets.  
- Can be applied to precision/recall if sample sizes are large.  
- Not recommended for complex or bounded metrics (F1, ROC-AUC).  
- Usually replaced by **Wilson score** or **bootstrap** methods.

---

## 8. Summary
- Easiest CI method for ML metrics.  
- Good as a teaching tool and quick baseline.  
- **Not reliable for small or imbalanced datasets.**  
- Prefer Wilson (Method 2) or Bootstrap (later methods) for serious use.


# Method 2 — Wilson Score Confidence Interval

## 1. What it is
The **Wilson Score interval** is a better alternative to the Normal/Wald CI for proportions.  
It adjusts the interval using the Normal approximation **plus a correction term** that keeps the bounds within [0,1] and improves coverage, especially for small or imbalanced samples.

Formula (two-sided CI):

$$
CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

Where:
- $\hat{p}$ = observed proportion (e.g., accuracy)  
- $n$ = number of samples  
- $z$ = critical Normal value (≈1.96 for 95%)  

---

## 2. How to compute
Example: **Accuracy = 0.84 on n = 500 samples**

1. Plug in values: $\hat{p} = 0.84$, $n = 500$, $z = 1.96$.  
2. Compute numerator terms (center and margin) and divide by the denominator $1 + z^2/n$.  
3. Result: interval ≈ $[0.807, 0.868]$.  
(This is very close to Wald’s $[0.81, 0.87]$, but Wilson has better coverage.)

---

## 3. When it works
- **Any binary classification metric that is a proportion**: accuracy, recall, precision, sensitivity, specificity.  
- Especially good when **$n$ is small** or the metric is near 0 or 1.  
- Recommended as the **default CI for proportions**.

---

## 4. Pros
- Much better coverage than Wald.  
- Interval stays within [0,1].  
- Handles small samples and extreme probabilities more reliably.  
- Still closed-form (fast, no resampling).

---

## 5. Cons
- Slightly more complex formula than Wald.  
- Still an approximation; may be conservative for very tiny samples ($n<10$).  
- Not directly applicable to complex metrics (F1, AUC, etc.).

---

## 6. Pitfalls
- Don’t confuse Wilson with Wald (many libraries still default to Wald).  
- For very small $n$, consider **Clopper–Pearson (exact)**.

---

## 7. Use in ML metrics
- Best choice for **accuracy** on small to medium test sets.  
- Works well for **precision/recall** treated as proportions.  
- For composite metrics (F1, AUC), bootstrap is preferred.

---

## 8. Summary
- Wilson Score Interval is a **major improvement** over Wald.  
- Closed-form, efficient, bounded in [0,1].  
- **Default recommendation for accuracy, precision, recall, sensitivity, specificity.**  
- For very small $n$ → Clopper–Pearson; for complex metrics → bootstrap.


# Method 3 — Clopper–Pearson (Exact) Confidence Interval

## 1. What it is
The **Clopper–Pearson interval** (often called the *exact binomial interval*) is a conservative method for proportions.  
It does not rely on Normal approximations — instead, it uses the **cumulative distribution of the Binomial** (or equivalently, the Beta distribution) to get the confidence limits.

Formula (two-sided interval):

$$
CI = \Big[ \text{BetaInv}\!\left(\tfrac{\alpha}{2}, k, n-k+1\right),\;
           \text{BetaInv}\!\left(1 - \tfrac{\alpha}{2}, k+1, n-k\right) \Big]
$$

Where:
- $k$ = number of successes (e.g., correct predictions)  
- $n$ = total number of samples  
- $\alpha$ = significance level (0.05 for 95% CI)  
- $\text{BetaInv}(q, a, b)$ = quantile function of the Beta distribution  

---

## 2. How to compute
Example: **Accuracy = 0.84 on $n = 500$ samples**

1. Number of successes: $k = 0.84 \times 500 = 420$.  
2. Lower bound: $\text{BetaInv}(0.025, 420, 81)$.  
3. Upper bound: $\text{BetaInv}(0.975, 421, 80)$.  
4. Result ≈ $[0.805, 0.869]$ (close to Wilson but with guaranteed coverage).

---

## 3. When it works
- Any **proportion metric**: accuracy, recall, precision, specificity.  
- Especially valuable when **$n$ is very small** (e.g., <30 samples).  
- Provides guaranteed coverage at the stated confidence level.

---

## 4. Pros
- **Exact coverage**: never underestimates uncertainty.  
- Works even for **very small $n$**.  
- Always bounded in [0,1].  

---

## 5. Cons
- Often **conservative**: intervals are wider than necessary.  
- Requires Beta quantiles (more compute than Wald/Wilson).  
- Not for complex metrics (F1, AUC, log-loss).

---

## 6. Pitfalls
- “Exact” ≠ “narrow” — it’s usually wider.  
- For large datasets, Wilson or bootstrap may be preferable (tighter).

---

## 7. Use in ML metrics
- Useful for **accuracy/recall** on very small test sets.  
- Strong choice for **medical ML** or **safety-critical domains**.  

---

## 8. Summary
- Clopper–Pearson is an **exact binomial interval**.  
- Guarantees coverage but can be wide.  
- Best for **tiny datasets or high-stakes applications**; otherwise prefer Wilson or bootstrap.


# Method 4 — Agresti–Coull Confidence Interval

## 1. What it is
The **Agresti–Coull interval** is a modification of the Wald interval that improves coverage by adding “pseudo-counts” to the sample.  
It is often called the **add-2 method** (for 95% CIs), because it adds 2 successes and 2 failures to stabilize the estimate.

Formula (two-sided 95% CI):

1. Adjusted sample size:

$$
n' = n + z^2
$$

2. Adjusted proportion:

$$
\tilde{p} = \frac{k + \tfrac{z^2}{2}}{n'}
$$

3. Interval:

$$
CI = \tilde{p} \pm z \sqrt{ \frac{\tilde{p}(1 - \tilde{p})}{n'} }
$$

Where:
- $k$ = number of successes (e.g., correct predictions)  
- $n$ = total number of samples  
- $z$ = critical Normal value (≈1.96 for 95%)  

---

## 2. How to compute
Example: **Accuracy = 0.84 on $n = 500$ samples**

1. Compute adjusted values:  
   - $n' = 500 + 1.96^2 \approx 503.84$  
   - $\tilde{p} = (420 + 1.96^2/2)/503.84 \approx 0.837$

2. Standard error:

$$
SE = \sqrt{\frac{0.837 (1 - 0.837)}{503.84}} \approx 0.016
$$

3. CI:

$$
0.837 \pm 1.96 \cdot 0.016 = [0.805, 0.869]
$$

---

## 3. When it works
- Proportion metrics (accuracy, recall, precision).  
- **Small/medium $n$** where Wald fails.  
- Coverage close to Wilson but easier to compute.

---

## 4. Pros
- Simple closed-form adjustment.  
- Better coverage than Wald.  
- Bounded in [0,1].  

---

## 5. Cons
- Slightly wider than Wilson sometimes.  
- Still approximate (not exact).  
- Not recommended for very small $n$ (prefer exact).

---

## 6. Pitfalls
- Adjust both numerator and denominator.  
- For highly imbalanced metrics, bootstrap may still be better.

---

## 7. Use in ML metrics
- Good for **accuracy** reporting quickly and reliably.  
- Also works for **precision/recall** if counts aren’t tiny.  
- Not designed for composite metrics (F1, AUC).

---

## 8. Summary
- Agresti–Coull is a **smoothed Wald interval**.  
- Much better coverage than Wald, near Wilson.  
- A **simple, robust default** for proportions in ML.


# Method 5 — Jeffreys Confidence Interval

## 1. What it is
The **Jeffreys interval** is a Bayesian confidence interval (credible interval) for a binomial proportion.  
It uses the **Jeffreys prior**, a Beta$(0.5, 0.5)$ distribution, and then takes posterior quantiles as the CI.  
Well-calibrated even for small $n$, and less conservative than Clopper–Pearson.

Posterior distribution:

$$
\theta \mid k,n \sim \text{Beta}(k+0.5,\, n-k+0.5)
$$

CI bounds (two-sided, $1-\alpha$):

$$
CI = \Big[ \text{BetaInv}\!\big(\tfrac{\alpha}{2}, k+0.5, n-k+0.5 \big),\;
           \text{BetaInv}\!\big(1 - \tfrac{\alpha}{2}, k+0.5, n-k+0.5 \big) \Big]
$$

---

## 2. How to compute
Example: **Accuracy = 0.84 on $n = 500$ samples**

1. $k = 420$, $n = 500$.  
2. Posterior = Beta$(420.5, 80.5)$.  
3. 2.5% and 97.5% quantiles → interval ≈ $[0.807, 0.869]$.

---

## 3. When it works
- Any **binomial proportion** metric.  
- Especially strong for **small $n$**.  
- Balanced between exact conservatism and approximation.

---

## 4. Pros
- Well-calibrated for small samples.  
- Bounded in [0,1].  
- Usually tighter than Clopper–Pearson.  
- Bayesian interpretation (credible interval).

---

## 5. Cons
- Needs Beta quantiles.  
- Bayesian interpretation may not be desired by all readers.  
- Not directly applicable to F1, AUC, etc.

---

## 6. Pitfalls
- Don’t confuse with Wald — Jeffreys uses a **prior**.  
- With very tiny $n$, intervals remain wide (unavoidable).

---

## 7. Use in ML metrics
- Good for **accuracy/precision/recall** when $n$ is small.  
- Preferred over Wald/Agresti–Coull when Bayesian framing is acceptable.

---

## 8. Summary
- Jeffreys = **Bayesian credible interval** with Beta$(0.5,0.5)$ prior.  
- Good coverage without being overly conservative.  
- Use for **proportion metrics** with small or moderate $n$.


# Method 6 — Likelihood-Ratio Confidence Interval

## 1. What it is
The **Likelihood-Ratio (LR) interval** is based on inverting a binomial likelihood ratio test.  
It finds values of $\theta$ for which the likelihood ratio matches the chosen significance level.

For $k$ successes in $n$ trials:

$$
\Lambda(\theta) = \frac{L(\theta)}{L(\hat{p})}
= \frac{ \binom{n}{k} \theta^k (1-\theta)^{n-k} }{ \binom{n}{k} \hat{p}^k (1-\hat{p})^{n-k} }
$$

with $\hat{p}=k/n$. The confidence set is all $\theta$ such that

$$
-2 \ln \Lambda(\theta) \le \chi^2_{1,\,1-\alpha}.
$$

---

## 2. How to compute
Example: **Accuracy = 0.84 on $n = 500$**

1. $k = 420$, $n = 500$, $\hat{p} = 0.84$.  
2. For candidate $\theta$, compute $-2 \ln \Lambda(\theta)$.  
3. Keep all $\theta$ where the statistic $\le \chi^2_{1,0.95} \approx 3.84$.  
4. Numerical root-finding gives bounds.  
5. Result ≈ $[0.807, 0.869]$.

---

## 3. When it works
- Proportion metrics (accuracy, recall, precision).  
- **Small to medium $n$** where Wald fails.  
- Accurate coverage without being overly conservative.

---

## 4. Pros
- Likelihood-based and principled.  
- Better calibration than Wald; close to Jeffreys.  
- Bounded within [0,1].

---

## 5. Cons
- Requires numerical root-finding.  
- More compute than Wald/Wilson.  
- Less common in standard ML libraries.

---

## 6. Pitfalls
- Care needed for extreme probabilities.  
- Numerical issues can give incorrect intervals.

---

## 7. Use in ML metrics
- Suitable for **accuracy** reporting with likelihood rigor.  
- Common in more statistical settings.

---

## 8. Summary
- LR interval via likelihood inversion.  
- Accurate but computationally heavier than simple formulas.  
- For large datasets, Wilson/Agresti–Coull are often sufficient.


# Method 7 — Bootstrap Percentile Confidence Interval

## 1. What it is
The **Bootstrap Percentile Interval** estimates the CI by **resampling the data** many times, recomputing the metric, and taking empirical quantiles.

Steps:
1. Resample the test set with replacement $B$ times.  
2. For each resample, compute the metric.  
3. Build the bootstrap distribution $\{m_1,\dots,m_B\}$.  
4. CI = empirical quantiles:

$$
CI = \big[\, q_{\alpha/2},\; q_{1-\alpha/2} \,\big]
$$

---

## 2. How to compute
Example: **Accuracy = 0.84 on $n = 500$**

1. Choose $B = 2000$.  
2. Compute accuracy on each resample.  
3. CI = 2.5th and 97.5th percentiles.  
4. Result ≈ $[0.808, 0.870]$.

---

## 3. When it works
- **Any metric**: accuracy, F1, AUC, PR-AUC, log-loss, RMSE, etc.  
- Good for nonlinear, skewed, or bounded metrics.

---

## 4. Pros
- General-purpose, model-agnostic.  
- Works for complex ML metrics.  
- Simple to explain and visualize.

---

## 5. Cons
- May **under-cover** if distribution is skewed.  
- Sensitive with very small $n$.  
- Computationally heavier (many resamples).  

---

## 6. Pitfalls
- Resample **indices** (keep features/labels paired).  
- Use **stratified resampling** for imbalance.  
- Use enough resamples ($\ge 1000$, preferably $5000+$).  
- Use block/cluster bootstrap for dependent data.

---

## 7. Use in ML metrics
- Great for **F1**, **PR-AUC**, **ROC-AUC** (when not using DeLong).  
- Regression metrics (RMSE, MAE, $R^2$).  
- Comparing two models via paired bootstrap.

---

## 8. Summary
- Simple, widely used bootstrap CI.  
- May under-cover for skew; BCa is more accurate.


# Method 8 — Bootstrap BCa (Bias-Corrected and Accelerated) Confidence Interval

## 1. What it is
**BCa** improves the percentile CI by correcting for **bias** ($z_0$) and **skewness** (acceleration $a$):

Conceptually,

$$
CI = \big[\, q_{\alpha_1},\; q_{\alpha_2} \,\big],
$$

where the adjusted quantile levels are

$$
\alpha_1 = \Phi \!\Big( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a (z_0 + z_{\alpha/2})} \Big),\qquad
\alpha_2 = \Phi \!\Big( z_0 + \frac{z_0 + z_{1-\alpha/2}}{1 - a (z_0 + z_{1-\alpha/2})} \Big).
$$

---

## 2. How to compute
Example: **F1-score**

1. Generate $B=5000$ bootstrap replicates.  
2. Compute $z_0$ (bias) and $a$ (via jackknife).  
3. Use adjusted quantile levels to read CI from the bootstrap distribution.  
4. Interval often asymmetric vs. percentile CI.

---

## 3. When it works
- **Any ML metric**; especially skewed ones (F1, PR-AUC).  
- When precision matters.

---

## 4. Pros
- Better coverage than percentile.  
- Corrects for bias and skew.  
- Robust for bounded, non-Normal metrics.

---

## 5. Cons
- More compute (bootstrap + jackknife).  
- Implementation details matter.

---

## 6. Pitfalls
- Stratify for imbalance.  
- Jackknife acceleration unstable for very small $n$.

---

## 7. Use in ML metrics
- Excellent for **F1** and **PR-AUC**.  
- Preferred for model comparisons.  
- Good for skewed regression errors.

---

## 8. Summary
- BCa is the **most reliable bootstrap CI** for ML metrics.  
- Heavier compute but worth it when accuracy matters.


# Method 9 — Bootstrap Normal Confidence Interval

## 1. What it is
Uses bootstrap to estimate **SE** and then applies a Normal-style CI:

$$
CI = \hat{\theta} \pm z_{1-\alpha/2} \cdot \widehat{SE}_{\text{boot}},
\qquad
\widehat{SE}_{\text{boot}} = \text{sd}(m_1,\dots,m_B).
$$

---

## 2. How to compute
Example: **AUC**

1. Compute observed AUC = 0.92.  
2. Generate $B=2000$ bootstrap AUCs.  
3. $CI = 0.92 \pm 1.96 \cdot \widehat{SE}_{\text{boot}}$.

---

## 3. When it works
- Any metric with approximately **symmetric** bootstrap distribution.

---

## 4. Pros
- Simple.  
- Faster than BCa (no jackknife).

---

## 5. Cons
- Assumes symmetry; can exceed bounds for [0,1] metrics.  
- Less accurate than percentile/BCa when skewed.

---

## 6. Pitfalls
- Avoid when histogram is skewed.  

---

## 7. Use in ML metrics
- OK for AUC, F1, precision/recall if distribution is symmetric.

---

## 8. Summary
- Easy approximation; BCa usually preferred for publication.


# Method 10 — Jackknife-Normal Confidence Interval

## 1. What it is
Leave-one-out (jackknife) estimates to get SE, then Normal CI.

Steps:

$$
\hat{\theta}_{(i)} = m(D \setminus \{i\}),\quad
\bar{\theta}_{\text{jack}} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_{(i)},\quad
\widehat{Var}_{\text{jack}} = \frac{n-1}{n}\sum_{i=1}^n(\hat{\theta}_{(i)}-\bar{\theta}_{\text{jack}})^2,
$$

$$
CI = \hat{\theta} \pm z_{1-\alpha/2}\cdot \sqrt{\widehat{Var}_{\text{jack}}}.
$$

---

## 2. How to compute
Example: **F1-score on $n=100$**

- Compute $n$ leave-one-out estimates, then SE and CI.

---

## 3. When it works
- **Smooth** statistics (means, regression coefficients).  
- Less reliable for **non-smooth** metrics (e.g., F1).

---

## 4. Pros
- Deterministic.  
- Only $n$ recomputations.

---

## 5. Cons
- Less accurate for skewed/nonlinear metrics.  
- Can extend outside valid range.

---

## 6. Pitfalls
- Non-smooth metrics can misbehave.  

---

## 7. Use in ML metrics
- Better for regression metrics (RMSE, MAE) than classification ones.

---

## 8. Summary
- Jackknife-Normal: simple, deterministic; not robust for non-smooth metrics.


# Method 11 — DeLong’s Confidence Interval for ROC-AUC

## 1. What it is
**DeLong’s method** gives a nonparametric, closed-form CI for ROC-AUC using **U-statistics**.

Variance:

$$
\widehat{Var}(\widehat{AUC}) = \frac{1}{n_+}\, Var_x(\phi(x)) + \frac{1}{n_-}\, Var_y(\psi(y)).
$$

CI:

$$
CI = \widehat{AUC} \pm z_{1-\alpha/2}\cdot \sqrt{\widehat{Var}(\widehat{AUC})}.
$$

---

## 2. How to compute
Example: **ROC-AUC = 0.92, $n_+=200$, $n_-=300$**

- Use implementations (e.g., `pROC` in R, Python ports) to get SE and CI.  
- Result ≈ $[0.89, 0.95]$.

---

## 3. When it works
- Only for **ROC-AUC** (binary).  
- Best for moderate/large $n$.

---

## 4. Pros
- Closed-form, efficient, nonparametric.  
- Standard in biomedical ML.

---

## 5. Cons
- Only AUC; assumes i.i.d. samples.  
- May underestimate variance for very small $n$.

---

## 6. Pitfalls
- Use paired DeLong when comparing two models.  
- Consider resampling for clustered/imbalanced data.

---

## 7. Use in ML metrics
- **Go-to** for ROC-AUC CI.

---

## 8. Summary
- Prefer DeLong for AUC over bootstrap when applicable.


# Method 12 — Nadeau–Bengio Correction (Cross-Validation Confidence Interval)

## 1. What it is
CI for metrics estimated via **$k$-fold cross-validation**, correcting variance inflation due to overlapping training sets.

Variance correction:

$$
\widehat{Var}_{NB} = \frac{1}{k}\widehat{\sigma}^2 + \frac{1}{k-1}\widehat{\sigma}^2,
$$

or with $r$ repeats and $k$ folds,

$$
\widehat{Var}_{NB} = \frac{1}{rk}\widehat{\sigma}^2 + \frac{1}{r(k-1)}\widehat{\sigma}^2.
$$

CI:

$$
CI = \bar{\theta} \pm t_{1-\alpha/2,\,df}\cdot \sqrt{\widehat{Var}_{NB}}.
$$

---

## 2. How to compute
Example: **10-fold CV**

1. Collect fold scores $\theta_1,\dots,\theta_{10}$.  
2. Compute mean $\bar{\theta}$ and variance $\widehat{\sigma}^2$.  
3. Apply NB correction; use $t$-quantile ($df=k-1$).  
4. CI is **wider** than naïve CV CI.

---

## 3. When it works
- CV-based performance estimates.  
- Any metric (classification or regression).

---

## 4. Pros
- Corrects dependence between folds.  
- Widely cited; more honest than naïve CV.

---

## 5. Cons
- Conservative.  
- Only for CV-based evaluation.

---

## 6. Pitfalls
- Don’t treat folds as independent.  
- Very small $k$ can be unstable.

---

## 7. Use in ML metrics
- Reporting **cross-validated** performance with uncertainty.

---

## 8. Summary
- Prefer NB over naïve CV intervals; for repeated CV, see Bouckaert–Frank.


# Method 13 — Bouckaert–Frank Correction (Repeated Cross-Validation Confidence Interval)

## 1. What it is
Refines NB for **repeated CV** ($r\times k$). Accounts for dependence across repetitions.

Variance correction:

$$
\widehat{Var}_{BF} = \frac{1}{rk}\widehat{\sigma}^2 + \frac{1}{r}\cdot \frac{1}{k-1}\widehat{\sigma}^2.
$$

CI:

$$
CI = \bar{\theta} \pm t_{1-\alpha/2,\,df}\cdot \sqrt{\widehat{Var}_{BF}}.
$$

---

## 2. How to compute
Example: **$10\times10$ repeated CV**

- Use all $rk=100$ scores to compute $\bar{\theta}$ and $\widehat{\sigma}^2$, then apply BF.  
- CI is **narrower** than NB but still wider than naïve CV.

---

## 3. When it works
- Repeated $k$-fold CV performance estimates.

---

## 4. Pros
- More efficient than NB for repeated CV.  
- Handles variance within folds **and** across repetitions.

---

## 5. Cons
- More complex; still approximate.  

---

## 6. Pitfalls
- Don’t average per repetition and treat as independent; apply the proper correction.  
- Use enough repetitions ($r\ge 5$) for stability.

---

## 7. Use in ML metrics
- Best for **repeated CV** results (accuracy, F1, AUC, regression errors).

---

## 8. Summary
- Bouckaert–Frank handles repeated CV dependence; recommended for $r\times k$ CV reports.


# 🔹 Advanced Extensions (Beyond the 13 Core Methods)

The 13 methods above cover almost all practical cases for ML evaluation.  
However, there are several **advanced or specialized approaches** that may be useful in specific contexts:

---

## 1. Permutation-based Confidence Intervals
- Construct CIs by **reshuffling labels** (permutation test).  
- Useful for **hypothesis testing** (e.g., “is model better than random?”).  
- Helpful in **model comparison**.

---

## 2. Bayesian Posterior Intervals
- Place a **prior** on metrics instead of bootstrapping.  
- Examples: Beta–Binomial for accuracy; Dirichlet–Multinomial for confusion matrices.  
- Produces **credible intervals** with Bayesian interpretation.

---

## 3. Block or Clustered Bootstrap
- Standard bootstrap assumes **i.i.d.**  
- For **time-series / grouped** data, use block or cluster bootstrap to preserve dependence.

---

## 4. Cross-Validated t-Test Variants
- For model comparisons with CV:  
  - Dietterich’s **$5\times2$cv** test.  
  - **Corrected resampled $t$-test** for repeated CV.

---

## 5. Asymptotic Delta Method
- Taylor expansion to approximate variance of nonlinear metrics (log-loss, Brier score, $R^2$).  
- Fast analytic CIs for large samples.

---

## 6. Bayesian Bootstrap
- Assign **Dirichlet** random weights to data points (no resampling rows).  
- Produces Bayesian-style bootstrap intervals.

---

## 7. Nested / Double Bootstrap
- Bootstrap within bootstrap; improves coverage in small samples / model selection.  
- Very computationally heavy.

---

# ✅ Summary
- These methods are **beyond the core 13**, but important for **special cases**:
  - **Permutation / Bayesian** → model comparison & small-sample analysis.  
  - **Block Bootstrap** → dependent data.  
  - **CV-corrected t-tests** → model comparison in CV studies.  
  - **Delta Method** → fast approximation for smooth metrics.  
  - **Bayesian & Nested Bootstrap** → advanced resampling with better theoretical properties.  

For most ML applications, the **13 main methods** are sufficient.  
These extensions are useful when working with **special data structures or research-heavy applications**.


# 📊 Summary of CI Evaluation Methods for ML Metrics

| Method | Type | Formula Style | Pros | Cons | Best For |
|--------|------|---------------|------|------|----------|
| **1. Wald (Normal)** | Analytic | Normal approx | Easiest, fast, simple | Poor coverage, invalid bounds | Large n, rough baseline for accuracy |
| **2. Wilson Score** | Analytic | Adjusted Normal | Better coverage, bounded [0,1] | Slightly more complex formula | Accuracy, precision, recall (default choice) |
| **3. Clopper–Pearson (Exact)** | Analytic | Beta quantiles | Exact coverage, tiny n | Conservative (wide CIs) | Small n, high-stakes domains |
| **4. Agresti–Coull** | Analytic | Wald + pseudo-counts | Simple, robust, bounded | Slightly wider than Wilson | Accuracy/precision on small–medium n |
| **5. Jeffreys** | Bayesian | Beta(0.5,0.5) posterior | Well-calibrated, shorter than exact | Needs Beta quantiles | Proportions with small n |
| **6. Likelihood-Ratio** | Analytic/Numerical | Likelihood inversion | Rigorous, bounded | Needs root-finding | Proportions in research settings |
| **7. Bootstrap Percentile** | Resampling | Empirical quantiles | General-purpose | May under-cover if skewed | F1, AUC, PR-AUC, RMSE |
| **8. Bootstrap BCa** | Resampling | Bias + skew corrected | Best coverage | Heavier compute | Default for complex metrics |
| **9. Bootstrap Normal** | Resampling | Wald with boot SE | Simple, faster than BCa | Assumes symmetry, can exceed bounds | Symmetric metric distributions |
| **10. Jackknife-Normal** | Resampling | Leave-one-out SE | Deterministic, $n$ runs | Bad for non-smooth metrics | Regression metrics (RMSE, MAE) |
| **11. DeLong (ROC-AUC)** | Analytic (U-statistics) | AUC variance | Standard for AUC | Only AUC, i.i.d. | ROC-AUC in ML & biomedicine |
| **12. Nadeau–Bengio (CV)** | CV correction | Variance inflation | Corrects fold dependence | Conservative | Single $k$-fold CV metrics |
| **13. Bouckaert–Frank (Repeated CV)** | CV correction | Repeated CV variance | Efficient for repeated CV | Complex, many reps | Repeated CV (e.g., $10\times10$) |
| **Advanced Extensions** | Specialized | – | Handle special cases | Complex/rare | Time series, small n, Bayesian ML, comparisons |


# 📋 Comprehensive CI Method Guide for ML Metrics

| Situation / Context            | Metric Family / Examples        | Recommended CI Method | Why / Best Use Case | Pros | Cons / Warnings |
|--------------------------------|---------------------------------|------------------------|---------------------|------|-----------------|
| **Small n (<30) or extreme prevalence (p≈0 or 1)** | Accuracy, Recall, Precision, Specificity, PICP | **Jeffreys** (preferred) or **Clopper–Pearson** | Handles boundary issues; exact coverage | Jeffreys less conservative, CP exact | CP overly conservative; wider intervals |
| **Holdout, decent n**          | Accuracy, Recall, Precision, Specificity, PICP | **Wilson score interval** | Robust alternative to Wald | Good coverage, shorter intervals than CP | Avoid Wald; fails at boundaries |
| **ROC-AUC (single model)**     | Binary classification AUC       | **DeLong** | Analytic variance via U-statistics | Fast, exact (no resampling) | Needs continuous scores, can struggle with heavy ties |
| **Δ ROC-AUC (paired models)**  | Compare two models on same items | **Paired DeLong** | Exploits covariance of scores | Efficient, widely used | Assumes paired setting only |
| **PR-AUC / Average Precision** | Imbalanced classification       | **Stratified BCa bootstrap** | No analytic formula exists | Handles class imbalance, bias-corrected | Must ensure positives present in resamples |
| **F1 / Fβ / MCC / κ / Macro**  | F1, MCC, Cohen’s κ, macro-averaged metrics | **Stratified BCa bootstrap** | Skewed ratio metrics | Bias-corrected, robust | Computationally heavy |
| **Regression (light-tailed)**  | MSE, RMSE                       | **Normal approximation** (delta/Studentized) | Residuals approx. normal | Simple, analytic | Misleading if residuals skewed/heavy-tailed |
| **Regression (heavy-tailed)**  | MAE, MAPE, MdAE, R²             | **BCa bootstrap** | Robust to skew/heavy tails | Bias-corrected, skew-aware | Slower, needs many resamples |
| **Ranking / IR metrics**       | NDCG, MRR, Hit@k, MAP           | **Query-level BCa bootstrap** | Metrics defined per-query | Respects query dependence | Needs enough queries to resample |
| **Calibration metrics**        | Brier score, ECE, MCE           | **Bootstrap** (studentized > BCa) | Rebinning at each resample | Handles non-linear calibration metrics | Studentized bootstrap harder to implement |
| **Time-series / sequential**   | Forecast errors, PICP           | **Block bootstrap** (moving-block, ℓ ≈ n^(1/3)); for PICP also Wilson on effective n | Preserves autocorrelation | Corrects for dependence | Block length tuning matters |
| **Cross-validation (CV)**      | Any metric (fold means)         | **Nadeau–Bengio** or **Bouckaert–Frank** corrected t-interval | Corrects variance under overlap | Standard in ML eval literature | Needs correct correction factor |
| **Paired Δ Accuracy**          | Accuracy difference (same items)| **McNemar CI** | Standard for paired binary outcomes | Exact, interpretable | Limited to binary outcomes |
| **Paired Δ (other metrics)**   | F1, MCC, κ, etc.                | **Paired BCa bootstrap** | Captures item-level pairing | Flexible, non-parametric | Expensive if dataset large |
| **Unpaired Δ (different items)** | Any metric                    | **Two-sample bootstrap** (or delta method if smooth/normal) | Models on different datasets | Easy to apply | Larger variance, independence assumed |
| **Default / fallback**         | Any metric (unsure case)        | **BCa bootstrap** | Safe robust choice when unsure | Works for most metrics | Computational cost |

---

## ✅ Quick Tips
- **Avoid Wald intervals** (unstable, misleading near boundaries).
- **Stratify bootstraps** when classes are imbalanced (PR-AUC, F1, Recall).
- **Check dependence**:  
  - Use **block bootstrap** for time-series.  
  - Use **query/cluster bootstrap** for grouped data.  
- **Cross-validation**: Never report naïve SE across folds → always use **NB/BF correction**.
- **Paired comparisons**: Exploit pairing (McNemar, Paired DeLong, Paired Bootstrap).  
- **Small samples**: Use **Jeffreys** or **Clopper–Pearson**, never Wilson/Wald.

---



# 📚 References & Resources

## Classic Foundations
- Clopper, C.J. & Pearson, E.S. (1934). *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial*. **Biometrika**.  
- Jeffreys, H. (1946). *An Invariant Form for the Prior Probability in Estimation Problems*. **Proc. Royal Society A**.  
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. **Annals of Statistics**.  
- Agresti, A. & Coull, B.A. (1998). *Approximate is Better than “Exact” for Interval Estimation of Binomial Proportions*. **The American Statistician**.  
- DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). *Comparing the Areas under Two or More Correlated ROC Curves*. **Biometrics**.  
- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error*. **Machine Learning**.  
- Bouckaert, R.R. & Frank, E. (2004). *Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms*. **PKDD**.

---

## Modern Tutorials & Surveys
- Brown, L.D., Cai, T.T., & DasGupta, A. (2001). *Interval Estimation for a Binomial Proportion*. **Statistical Science**.  
- Newcombe, R.G. (1998). *Two-Sided Confidence Intervals for the Single Proportion: Comparison of Seven Methods*. **Statistics in Medicine**.  
- Kazmierczak et al. (2022). *Bootstrap Methods for Quantifying the Uncertainty of Experimental Data*. **Analytica Chimica Acta**.  
- MAPIE Project (2022). *Model Agnostic Prediction Interval Estimator*. **arXiv:2207.12274**.  
- Recent simulation studies (2024–2025) comparing bootstrap vs BCa in ML contexts (open-access articles).

---

## Open-Source Libraries & Repos
- **confidenceinterval** (Python): CI calculations for ML metrics.  
- **confidence-intervals** (PyPI): Lightweight CI computations including bootstrap methods.  
- **MAPIE** (scikit-learn-contrib): Prediction intervals & conformal inference.  
- **scipy.stats** / **statsmodels**: Proportion CIs, t-intervals, bootstrap utilities.  
- **pROC** (R): ROC analysis including DeLong’s CI.  
- **mlxtend** (Python): Bootstrap/permutation utilities.

---

## Practical Articles & Guides
- “Bootstrapping Confidence Intervals in Machine Learning” — blog tutorials with Python code.  
- “Confidence Intervals for Machine Learning Metrics” — GitHub/Kaggle notebooks.  
- StatQuest videos on confidence intervals (teaching intuition).
