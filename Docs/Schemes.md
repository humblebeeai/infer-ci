# Documentation: Evaluation Schemes for Confidence Intervals

When estimating the performance of machine learning models, the *evaluation scheme* defines how the data were split and how the metric was computed.  
Because different schemes introduce different types of statistical dependence in the results, the choice of confidence interval (CI) method must respect the scheme.  

This documentation explains the main evaluation schemes we support, the assumptions they make about the data, and the recommended CI methods for each.  

The goal is to provide a clear, structured reference:  
- **What the scheme is** (context and setup)  
- **How the data are structured** (independence, dependence, grouping, temporal order, etc.)  
- **Which CI methods are appropriate** (classical, bootstrap, corrected CV, block bootstrap, etc.)  
- **Examples** of how to call the unified `eval_ci(...)` interface for each scheme  

By following this guide, users can choose the correct CI method automatically based on the evaluation scheme, ensuring statistically valid uncertainty estimates for ML metrics across a wide variety of scenarios.


# Holdout (default)

**When:**  
- Train on training set, evaluate once on a separate test/validation set.  

**Data:**  
- Independent and identically distributed (i.i.d.) samples.  

**CI methods:**  
- Proportion metrics → Wilson, Jeffreys, Clopper–Pearson.  
- Nonlinear/skewed metrics (F1, κ, etc.) → Bootstrap (percentile/BCa).  

**Example:**
```python
res = eval_ci(y_true, y_pred, metric="accuracy", scheme="holdout")


# Cross-Validation (k-fold)

**When:**  
- Dataset is split into *k* folds.  
- Each fold is used once as validation, while the other k–1 folds are used for training.  
- The metric is averaged across the k validation folds.  

**Data:**  
- Fold estimates are *not independent* because the training sets overlap heavily.  
- Naïve standard error underestimates uncertainty.  

**CI methods:**  
- Use **Nadeau–Bengio corrected variance** formula.  
- This accounts for the dependence between folds and produces valid intervals.  

**Example:**
```python
res = eval_ci(
    y_true=None,
    y_pred=None,
    metric="accuracy",
    scheme="kfold",
    fold_scores=fold_scores,
    kfold_k=5
)


# Repeated Cross-Validation

**When:**  
- Cross-validation is repeated *r* times with different random splits.  
- Each run produces k validation folds, giving a total of r × k scores.  
- The metric is usually reported as the average over all folds.  

**Data:**  
- Even stronger dependence than plain k-fold CV:  
  - Folds within a run overlap.  
  - Folds across different runs also overlap.  
- Naïve variance severely underestimates uncertainty.  

**CI methods:**  
- Use **Bouckaert–Frank correction**, which extends the Nadeau–Bengio method to repeated CV.  
- This accounts for both within-run and across-run dependence.  

**Example:**
```python
res = eval_ci(
    y_true=None,
    y_pred=None,
    metric="accuracy",
    scheme="repeated_cv",
    fold_scores=fold_scores,  # length = r × k
    kfold_k=5,
    r_repeats=3
)


# Repeated Cross-Validation

**When:**  
- Cross-validation is repeated *r* times with different random splits.  
- Each run produces k validation folds, giving a total of r × k scores.  
- The metric is usually reported as the average over all folds.  

**Data:**  
- Even stronger dependence than plain k-fold CV:  
  - Folds within a run overlap.  
  - Folds across different runs also overlap.  
- Naïve variance severely underestimates uncertainty.  

**CI methods:**  
- Use **Bouckaert–Frank correction**, which extends the Nadeau–Bengio method to repeated CV.  
- This accounts for both within-run and across-run dependence.  

**Example:**
```python
res = eval_ci(
    y_true=None,
    y_pred=None,
    metric="accuracy",
    scheme="repeated_cv",
    fold_scores=fold_scores,  # length = r × k
    kfold_k=5,
    r_repeats=3
)


# Time-Series (Blocked / Rolling CV)

**When:**  
- Data are sequential and exhibit temporal dependence (e.g., forecasting, stock prices, sensor data).  
- Validation is done on future segments, not random shuffles.  
- Common setups:  
  - **Blocked CV:** train on first block(s), test on next block.  
  - **Rolling CV:** gradually expand the training window and test on the next time step or block.  

**Data:**  
- Residuals/errors are *autocorrelated*.  
- Standard bootstrap or i.i.d. assumptions do not hold.  

**CI methods:**  
- Use **Block Bootstrap** (or Moving Block Bootstrap):  
  - Resample contiguous blocks of observations.  
  - Preserves autocorrelation structure.  
- Alternative: Circular Block Bootstrap or Stationary Bootstrap (advanced).  

**Example:**
```python
res = eval_ci(
    y_true=y_true,
    y_pred=y_pred,
    metric="rmse",
    scheme="ts_blocked",
    B=2000,
    block_length=None,  # defaults to n^(1/3) if not given
    seed=42
)


# Stratified Bootstrap (Imbalanced Data)

**When:**  
- Dataset has a strong **class imbalance** (e.g., 90% vs. 10%).  
- Metrics like F1, recall for the minority class, or precision are very sensitive to imbalance.  
- Standard bootstrap may create resamples that miss the minority class entirely → degenerate metrics.  

**Data:**  
- Classes are not equally represented.  
- Resampling must preserve the class distribution to avoid biased estimates.  

**CI methods:**  
- Use **Stratified Bootstrap**:  
  - Resample *within each class separately*, then combine.  
  - Ensures each bootstrap sample reflects the original imbalance.  
- Often paired with **BCa bootstrap** for skewed metrics (F1, MCC, κ).  

**Example:**
```python
res = eval_ci(
    y_true=y_true,
    y_pred=y_pred,
    metric="f1",
    scheme="holdout",
    imbalance="high",
    stratify=y_true,   # required for stratified bootstrap
    B=3000,
    seed=7
)


# Ranking / IR Metrics (Query-level Bootstrap)

**When:**  
- Metrics like **NDCG, MAP, MRR, Hit@k, Precision@k** are used in **information retrieval / ranking tasks**.  
- Data are grouped by queries (user → list of candidate items).  
- Scores are aggregated across queries.  

**Data:**  
- Items **within the same query are not independent** (they share the same user/query context).  
- Resampling items individually breaks this dependency.  

**CI methods:**  
- Use **Query-level Bootstrap**:  
  - Resample *queries* (rows), not items.  
  - Preserves within-query structure.  
- Recommended variant: **BCa bootstrap**, which adjusts for bias/skew in ranking metrics.  

**Example:**
```python
# y_true and y_score are 2D arrays: (n_queries × n_items)
res = eval_ci(
    y_true=y_true_queries,
    y_pred=None,
    metric="ndcg",
    scheme="holdout",
    y_score=y_score_queries,
    B=2000,
    seed=22
)


# Calibration Metrics (ECE, MCE, Brier)

**When:**  
- Evaluating **probabilistic predictions** (e.g., classifiers that output probabilities).  
- Metrics include:  
  - **ECE (Expected Calibration Error)**  
  - **MCE (Maximum Calibration Error)**  
  - **Brier Score**  

**Data:**  
- Calibration metrics depend on **binning predicted probabilities**.  
- Each resample must recompute the bins to avoid bias.  
- Metrics are nonlinear functions of bin frequencies → standard normal approximations fail.  

**CI methods:**  
- Use **Bootstrap (preferably BCa)**:  
  - Resample observations, recompute bins, then metric.  
  - BCa adjusts for bias and skewness.  
- Alternative: Studentized bootstrap for more accuracy.  

**Example:**
```python
res = eval_ci(
    y_true=y_true,
    y_pred=None,
    metric="ece",       # or "mce" or "brier"
    scheme="holdout",
    y_score=y_prob,     # predicted probabilities
    B=2000,
    seed=7
)


# Summary: Evaluation Schemes and CI Methods

| Scheme                | When it’s used                                        | Data structure / Dependence        | Recommended CI method(s)                           |
|------------------------|-------------------------------------------------------|------------------------------------|---------------------------------------------------|
| **Holdout**            | Single train/test split                               | i.i.d. samples                     | Wilson / Jeffreys / Clopper–Pearson (proportions), BCa bootstrap (skewed metrics) |
| **k-fold CV**          | Cross-validation with k folds                         | Folds overlap → dependent          | Nadeau–Bengio correction                          |
| **Repeated CV**        | Cross-validation repeated r times                     | Even stronger dependence           | Bouckaert–Frank correction                        |
| **Time-Series (Blocked)** | Forecasting, sequential validation                 | Temporal autocorrelation           | Block Bootstrap (resample contiguous blocks)      |
| **Stratified Bootstrap** | Imbalanced classification (e.g. 90/10)              | Class imbalance                    | Stratified BCa bootstrap                          |
| **Ranking / IR**       | Metrics like NDCG, MAP, MRR, Hit@k, Precision@k       | Grouped by queries (within-query dep.) | Query-level BCa bootstrap                     |
| **Calibration**        | Probabilistic metrics (ECE, MCE, Brier)               | Bin-based, nonlinear               | BCa bootstrap (recompute bins per resample)       |
