# Machine Learning Metrics Handbook

This handbook provides a structured overview of evaluation metrics used across **all domains of Machine Learning**.  
Each metric is explained with **definition, formula, advantages, limitations, and use cases**.  
The goal is to create a clear, practical reference for selecting the right metric in ML projects.

---

# 📑 Table of Contents

1. [Regression Metrics](#regression-metrics)  
   - MAE, MSE, RMSE, R², Adjusted R², MAPE, sMAPE, MedAE, Huber, MSLE, etc.  

2. [Classification Metrics](#classification-metrics)  
   - Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, Log-Loss, MCC, Cohen’s Kappa, Balanced Accuracy, etc.  

3. [Ranking & Information Retrieval Metrics](#ranking--information-retrieval-metrics)  
   - Precision@k, Recall@k, MAP, NDCG, MRR, Hit Ratio, Coverage, Diversity, Novelty, ERR, PRBEP, etc.  

4. [Clustering Metrics](#clustering-metrics)  
   - Silhouette Score, Davies–Bouldin, Calinski–Harabasz, Dunn Index, ARI, NMI, Purity, V-Measure, Entropy, Gap Statistic, Modularity, AIC/BIC, etc.  

5. [Time-Series Forecasting Metrics](#time-series-forecasting-metrics)  
   - MAE, MSE, RMSE, RMSPE, RMSSE, MSLE, MAPE, sMAPE, WAPE, MAAPE, MASE, Theil’s U, Pinball Loss, PICP, Interval Score, etc.  

6. [Computer Vision Metrics](#computer-vision-metrics)  
   - IoU, Dice, Pixel Accuracy, mIoU, Panoptic Quality, Boundary IoU, mAP, PSNR, SSIM, LPIPS, FID, IS, KID, etc.  

7. [Natural Language Processing Metrics](#natural-language-processing-nlp-metrics)  
   - BLEU, ROUGE, METEOR, GLEU, ChrF, TER, WER, CER, SPICE, COMET, QuestEval, BARTScore, BERTScore, MoverScore, Perplexity, EM, F1, etc.  

8. [Reinforcement Learning Metrics](#reinforcement-learning-rl-metrics)  
   - Cumulative Reward, Average Reward, Discounted Return, Success Rate, Regret, Sample Efficiency, Stability/Variance.  

9. [Generative Model Metrics](#generative-model-metrics)  
   - Inception Score, FID, KID, Mode Score, PR-GAN, Density & Coverage, Intra-FID, LPIPS, KDE Likelihood. 


- [📊 Master Summary Table of ML Metrics](#-master-summary-table-of-ml-metrics)  

---



# Regression Metrics

Regression metrics evaluate the performance of models that predict continuous numerical values, measuring the accuracy, error magnitude, and goodness-of-fit of predictions against actual outcomes.


# Mean Absolute Error (MAE)

## 1. What it is
The **Mean Absolute Error (MAE)** measures the average magnitude of errors between predicted values and actual values, without considering their direction (positive/negative).  
It is one of the most intuitive regression metrics, as it represents the average distance between predictions and ground truth.

## 2. Formula
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| y_i - \hat{y}_i \right|
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **MAE = 0** → Perfect predictions (no error).  
- The larger the MAE, the larger the prediction errors.  
- MAE is expressed in the same unit as the target variable (e.g., dollars, temperature).  

## 4. Advantages
- Simple to calculate and interpret.  
- Less sensitive to outliers compared to MSE/RMSE.  
- Directly corresponds to the average prediction error in real-world units.  

## 5. Limitations
- Does not penalize larger errors as strongly as MSE/RMSE.  
- Not differentiable at zero (can be a concern for optimization).  
- May be less suitable when very large errors must be heavily penalized.  

## 6. Use Cases
- Forecasting problems where interpretability matters (e.g., predicting house prices, energy consumption).  
- Business cases where average error in real-world units (e.g., \$, °C, kg) is more important than squaring large deviations.  


# Mean Squared Error (MSE)

## 1. What it is
The **Mean Squared Error (MSE)** measures the average of the squared differences between predicted values and actual values.  
By squaring the errors, MSE penalizes large deviations much more heavily than smaller ones, making it sensitive to outliers.

## 2. Formula
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **MSE = 0** → Perfect predictions.  
- A higher MSE indicates larger errors.  
- Because errors are squared, MSE exaggerates the impact of large errors.  

## 4. Advantages
- Strongly penalizes large errors (useful when large deviations are particularly undesirable).  
- Differentiable everywhere, making it suitable for gradient-based optimization.  
- Commonly used in regression tasks and theoretical analysis.  

## 5. Limitations
- Sensitive to outliers — a single large error can disproportionately increase the MSE.  
- Less interpretable than MAE, since errors are squared (units become squared too, e.g., dollars²).  

## 6. Use Cases
- Training regression models using least squares methods.  
- Scenarios where penalizing large errors is crucial (e.g., safety-critical predictions like energy load forecasting or medical applications).  
- As a loss function for optimization in machine learning (e.g., linear regression, neural networks).  


# Root Mean Squared Error (RMSE)

## 1. What it is
The **Root Mean Squared Error (RMSE)** is the square root of the Mean Squared Error (MSE).  
It measures the standard deviation of prediction errors (residuals), making it more interpretable than MSE because it brings the error back to the same unit as the target variable.

## 2. Formula
$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2 }
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **RMSE = 0** → Perfect predictions.  
- The higher the RMSE, the worse the model’s predictive performance.  
- RMSE is always greater than or equal to MAE, since squaring emphasizes larger errors.  

## 4. Advantages
- Same unit as the target variable, making it easier to interpret than MSE.  
- Sensitive to large errors, which is useful when big deviations are especially problematic.  
- Widely used and often reported as a standard regression metric.  

## 5. Limitations
- Still sensitive to outliers (even more than MAE).  
- Can be misleading if the dataset has extreme noise, since a few large errors can dominate the metric.  
- Harder to explain to non-technical audiences compared to MAE.  

## 6. Use Cases
- Common in forecasting tasks (e.g., weather prediction, energy demand).  
- Evaluating models where both average error and large deviations matter.  
- Benchmarking regression models in competitions (e.g., Kaggle, forecasting challenges).  


# Coefficient of Determination (R²)

## 1. What it is
The **Coefficient of Determination (R²)** measures how well the predicted values explain the variation of the actual values.  
It compares the model’s performance to a simple baseline model that always predicts the mean of the target variable.

## 2. Formula
$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( \bar{y} \) = mean of actual values  
- \( n \) = number of observations  

## 3. Interpretation
- \( R^2 = 1 \) → Perfect fit (all variance explained).  
- \( R^2 = 0 \) → Model is no better than simply predicting the mean.  
- \( R^2 < 0 \) → Model performs worse than predicting the mean.  

## 4. Advantages
- Intuitive interpretation: proportion of variance explained by the model.  
- Easy to compare across different models on the same dataset.  
- Widely used in regression analysis and academic research.  

## 5. Limitations
- Can be misleading for **non-linear models**.  
- Increases with more predictors, even if they are irrelevant (overfitting risk).  
- Not always meaningful for models that don’t minimize squared error (e.g., robust regressions).  

## 6. Use Cases
- Regression models where variance explanation is important.  
- Model comparison to check relative explanatory power.  
- Communicating performance to non-technical stakeholders (since it has a percentage-like interpretation).  


# Adjusted R²

## 1. What it is
The **Adjusted R²** is a modified version of R² that adjusts for the number of predictors in the model.  
Unlike R², which always increases (or at least never decreases) when adding more predictors, Adjusted R² introduces a penalty for including irrelevant variables.  

## 2. Formula
$$
\text{Adjusted } R^2 = 1 - \left(1 - R^2 \right) \cdot \frac{n - 1}{n - p - 1}
$$

Where:  
- \( R^2 \) = coefficient of determination  
- \( n \) = number of observations  
- \( p \) = number of predictors/features  

## 3. Interpretation
- Adjusted R² increases only if the new predictor improves the model more than would be expected by chance.  
- It can **decrease** if irrelevant predictors are added.  
- Helps prevent overfitting by discouraging unnecessary variables.  

## 4. Advantages
- Corrects the upward bias of R² when adding predictors.  
- More reliable metric for comparing models with different numbers of features.  
- Maintains interpretability (still variance explained, but adjusted).  

## 5. Limitations
- Still depends on linear regression assumptions (linearity, homoscedasticity).  
- Not meaningful for non-linear or non-parametric models.  
- Can be less intuitive to explain to non-technical audiences.  

## 6. Use Cases
- Comparing multiple regression models with different numbers of predictors.  
- Feature selection processes (choosing parsimonious models).  
- Academic and applied econometric research.  


# Mean Absolute Percentage Error (MAPE)

## 1. What it is
The **Mean Absolute Percentage Error (MAPE)** measures the average percentage difference between predicted values and actual values.  
It is widely used in forecasting because it expresses errors as percentages, making it easy to interpret across different scales.  

## 2. Formula
$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **MAPE = 0%** → Perfect predictions.  
- A MAPE of 10% means that predictions are off by 10% on average.  
- Lower is better.  

## 4. Advantages
- Easy to understand since it uses percentages.  
- Scale-independent (can compare across different datasets/units).  
- Popular in business and forecasting contexts.  

## 5. Limitations
- Undefined when \( y_i = 0 \) (division by zero issue).  
- Can be heavily biased when actual values are close to zero.  
- Asymmetric: over- and under-predictions impact the percentage differently.  

## 6. Use Cases
- Forecasting demand, sales, stock prices, or energy consumption.  
- Business and economics applications where percentage error is more meaningful than raw error.  
- Comparing models across datasets with different units.  


# Symmetric Mean Absolute Percentage Error (SMAPE)

## 1. What it is
The **Symmetric Mean Absolute Percentage Error (SMAPE)** is a modification of MAPE designed to avoid its main issues, especially division by zero and asymmetry between over- and under-predictions.  
It normalizes errors by the average of the actual and predicted values, making the percentage error more balanced.  

## 2. Formula
$$
\text{SMAPE} = \frac{100}{n} \sum_{i=1}^n \frac{\left| y_i - \hat{y}_i \right|}{\frac{|y_i| + |\hat{y}_i|}{2}}
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **SMAPE = 0%** → Perfect predictions.  
- Errors are expressed as percentages, but capped at **200%** due to the symmetric denominator.  
- More balanced between overestimation and underestimation compared to MAPE.  

## 4. Advantages
- Avoids division by zero when actual values are close to zero.  
- More symmetric in penalizing over- and under-predictions.  
- Useful when dealing with data that can take very small or zero values.  

## 5. Limitations
- Still less intuitive than MAPE for non-technical audiences.  
- Percentage interpretation is bounded (0–200%), which can be confusing.  
- Can still be sensitive to outliers.  

## 6. Use Cases
- Time series forecasting (sales, energy, demand).  
- Applications where actual values may be close to zero.  
- When a symmetric error metric is needed for fairness in over/under predictions.  


# Root Mean Squared Logarithmic Error (RMSLE)

## 1. What it is
The **Root Mean Squared Logarithmic Error (RMSLE)** measures the ratio-based differences between actual and predicted values by applying a logarithmic transformation.  
It is particularly useful when the target values vary across several orders of magnitude, or when relative differences are more important than absolute differences.  

## 2. Formula
$$
\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
$$

Where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of observations  

## 3. Interpretation
- **RMSLE = 0** → Perfect predictions.  
- Emphasizes **relative error**: predicting 10 instead of 20 is treated the same as predicting 100 instead of 200.  
- Penalizes **underestimation** more than overestimation.  

## 4. Advantages
- Handles large target ranges gracefully (log scale reduces the impact of very large values).  
- Reduces the influence of large outliers compared to RMSE.  
- Good for percentage growth-like problems (e.g., exponential growth).  

## 5. Limitations
- Requires non-negative target values (\( y_i \geq 0 \)).  
- Interpretation is less intuitive than MAE/RMSE.  
- Penalizes underestimation more than overestimation, which may not always be desired.  

## 6. Use Cases
- Forecasting problems where relative error is more important than absolute error (e.g., predicting population, sales growth, or number of users).  
- Kaggle competitions and ML benchmarks (RMSLE is often a scoring metric).  
- Business problems where percentage underestimation is more harmful than overestimation.  


# Median Absolute Error (MedAE)

## 1. What it is
The **Median Absolute Error (MedAE)** is the median of the absolute errors between actual and predicted values.  
Unlike MAE, which averages errors, MedAE takes the middle value, making it **more robust to outliers**.

## 2. Formula
$$
\text{MedAE} = \text{median}\left( |y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_n - \hat{y}_n| \right)
$$

## 3. Interpretation
- **MedAE = 0** → Perfect predictions.  
- Represents the typical (middle) error, rather than the average.  

## 4. Advantages
- Robust against extreme outliers.  
- Intuitive as a "typical error" measure.  

## 5. Limitations
- Ignores the size of the largest errors.  
- Not differentiable, so not used as a training loss.  

## 6. Use Cases
- Robust regression evaluation.  
- Situations where outliers should not dominate the error measure.  


# Huber Loss

## 1. What it is
The **Huber Loss** combines MAE and MSE:  
- It behaves like MSE for small errors (squared).  
- It behaves like MAE for large errors (absolute).  
This makes it robust to outliers while still being differentiable.

## 2. Formula
$$
L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

Where:  
- \( \delta \) = threshold parameter.  

## 3. Interpretation
- Small errors penalized quadratically.  
- Large errors penalized linearly.  

## 4. Advantages
- Robust to outliers (less than MSE).  
- Differentiable everywhere.  

## 5. Limitations
- Requires tuning parameter \( \delta \).  
- Less common for reporting, more for training.  

## 6. Use Cases
- Training regression models with noisy data.  
- Applications where robustness to outliers is needed.  


# Explained Variance Score

## 1. What it is
The **Explained Variance Score** measures the proportion of variance in the target variable that is captured by the predictions.  
It focuses on the variability of errors rather than their magnitude.

## 2. Formula
$$
\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
$$

Where:  
- \( y \) = actual values  
- \( \hat{y} \) = predicted values  
- \( \text{Var} \) = variance  

## 3. Interpretation
- Score = 1 → Perfect prediction (all variance explained).  
- Score = 0 → Predictions are as variable as the actuals.  
- Score < 0 → Predictions add more error than the baseline.  

## 4. Advantages
- Intuitive measure of how much variance is explained.  
- Related to R², but less affected by systematic bias.  

## 5. Limitations
- Can give high scores even with biased predictions (e.g., consistently underestimating).  
- Less interpretable to non-technical audiences.  

## 6. Use Cases
- Evaluating regression models when variance explanation is more important than raw error.  


# Mean Bias Deviation (MBE)

## 1. What it is
The **Mean Bias Deviation (MBE)** measures the **average bias** of predictions — whether the model tends to overestimate or underestimate.  
Unlike MAE, which takes the absolute value, MBE keeps the sign of the error.

## 2. Formula
$$
\text{MBE} = \frac{1}{n} \sum_{i=1}^n ( \hat{y}_i - y_i )
$$

## 3. Interpretation
- MBE > 0 → Model **overestimates** on average.  
- MBE < 0 → Model **underestimates** on average.  
- MBE ≈ 0 → Predictions are unbiased overall.  

## 4. Advantages
- Detects systematic bias in predictions.  
- Complements MAE/MSE by providing directional information.  

## 5. Limitations
- Positive and negative errors cancel each other out.  
- A small MBE does not necessarily mean small overall errors.  

## 6. Use Cases
- Energy forecasting, climate models, and finance (bias detection is crucial).  
- Model calibration checks.  


# Quantile Loss (Pinball Loss)

## 1. What it is
The **Quantile Loss (Pinball Loss)** is used when predicting quantiles instead of point estimates.  
It measures how well predicted quantiles match actual values, making it common in **quantile regression** and **uncertainty estimation**.

## 2. Formula
For a quantile \( q \in (0,1) \):
$$
L_q(y, \hat{y}) =
\begin{cases}
q \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(1 - q) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

## 3. Interpretation
- Penalizes underestimates and overestimates differently depending on quantile \( q \).  
- Example: \( q = 0.9 \) → focuses on upper-bound errors.  

## 4. Advantages
- Allows probabilistic forecasting (predicting intervals, not just averages).  
- Robust to skewed distributions.  

## 5. Limitations
- Less intuitive than MAE/MSE.  
- Requires choosing a quantile \( q \).  

## 6. Use Cases
- Forecasting with uncertainty (e.g., energy demand, stock price ranges).  
- Quantile regression and predictive intervals.  
- Applications where upper/lower bounds matter more than the average.  


# Mean Squared Logarithmic Error (MSLE)

## 1. What it is
The **Mean Squared Logarithmic Error (MSLE)** is similar to MSE, but it applies a logarithmic transformation before computing squared differences.  
It is useful when relative differences matter more than absolute ones.

## 2. Formula
$$
\text{MSLE} = \frac{1}{n} \sum_{i=1}^n \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2
$$

## 3. Interpretation
- MSLE = 0 → Perfect predictions.  
- Penalizes **underestimation** more than overestimation.  

## 4. Advantages
- Good for targets spanning several orders of magnitude.  
- Reduces the impact of very large values compared to MSE.  

## 5. Limitations
- Requires non-negative values (\( y_i \geq 0 \)).  
- Less intuitive than MAE/MSE.  

## 6. Use Cases
- Growth-related predictions (population, users, sales).  
- Kaggle competitions where log-scaled metrics are used.  


# Poisson Deviance

## 1. What it is
The **Poisson Deviance** is a regression metric for models with count data (Poisson-distributed targets).  
It compares the likelihood of the predicted model against the perfect model.

## 2. Formula
$$
D_{\text{Poisson}} = 2 \sum_{i=1}^n \left( y_i \cdot \log \frac{y_i}{\hat{y}_i} - (y_i - \hat{y}_i) \right)
$$

## 3. Interpretation
- Lower deviance → better fit.  
- Deviance = 0 → Perfect prediction.  

## 4. Advantages
- Suitable for **count data** (e.g., number of events).  
- Proper scoring rule (encourages honest probability predictions).  

## 5. Limitations
- Only valid when \( y_i \geq 0 \) and \( \hat{y}_i > 0 \).  
- Sensitive to zero or near-zero predictions.  

## 6. Use Cases
- Poisson regression models.  
- Predicting event counts (accidents, transactions, arrivals).  


# Gamma Deviance

## 1. What it is
The **Gamma Deviance** is used for positive, continuous, skewed data (Gamma-distributed targets).  
It is common in insurance, healthcare, and economics.

## 2. Formula
$$
D_{\text{Gamma}} = 2 \sum_{i=1}^n \left( \frac{y_i - \hat{y}_i}{\hat{y}_i} - \log \frac{y_i}{\hat{y}_i} \right)
$$

## 3. Interpretation
- Lower deviance → better fit.  
- Deviance = 0 → Perfect prediction.  

## 4. Advantages
- Suitable for **positive, skewed data**.  
- Works with generalized linear models (GLMs).  

## 5. Limitations
- Only valid when \( y_i > 0 \) and \( \hat{y}_i > 0 \).  
- Less intuitive to explain.  

## 6. Use Cases
- Insurance claim amounts.  
- Healthcare cost predictions.  
- Economics (skewed distributions).  


# Mean Tweedie Deviance

## 1. What it is
The **Tweedie Deviance** generalizes Poisson and Gamma Deviance, covering a family of distributions between them.  
It is controlled by a **power parameter** \( p \).

## 2. Formula
$$
D_{\text{Tweedie}}(p) = \frac{2}{n} \sum_{i=1}^n \left( \frac{y_i^{2-p}}{(1-p)(2-p)} - \frac{y_i \cdot \hat{y}_i^{1-p}}{1-p} + \frac{\hat{y}_i^{2-p}}{2-p} \right)
$$

Where:  
- \( p = 0 \) → Normal distribution (MSE).  
- \( p = 1 \) → Poisson distribution.  
- \( p = 2 \) → Gamma distribution.  
- \( 1 < p < 2 \) → Compound Poisson–Gamma (insurance, rainfall).  

## 3. Interpretation
- Lower deviance → better fit.  
- Useful when modeling skewed, heavy-tailed data.  

## 4. Advantages
- Flexible family of metrics.  
- Covers Normal, Poisson, Gamma, and in-between.  

## 5. Limitations
- Requires tuning parameter \( p \).  
- Interpretation is less straightforward.  

## 6. Use Cases
- Actuarial science and insurance.  
- Rainfall, energy consumption.  
- Any skewed positive data modeling.  


# Coefficient of Variation of RMSE (CV(RMSE))

## 1. What it is
The **Coefficient of Variation of RMSE (CV(RMSE))** normalizes RMSE by the mean of the actual values.  
This makes RMSE scale-independent, expressed as a percentage of the mean.

## 2. Formula
$$
\text{CV(RMSE)} = \frac{\text{RMSE}}{\bar{y}} \times 100
$$

Where:  
- \( \bar{y} \) = mean of actual values.  

## 3. Interpretation
- CV(RMSE) = 10% → RMSE is 10% of the average actual value.  
- Lower is better.  

## 4. Advantages
- Scale-independent (comparable across datasets).  
- Easy to interpret in percentage form.  

## 5. Limitations
- Undefined when mean of actuals = 0.  
- Sensitive to small mean values.  

## 6. Use Cases
- Energy and building performance forecasting.  
- Comparing regression models across different scales.  


# Classification Metrics

Classification metrics evaluate how well a model assigns inputs into discrete categories.  
They capture correctness, balance between false positives/false negatives, and the ability to separate classes — especially important when data is imbalanced.


# Accuracy

## 1. What it is
**Accuracy** is the proportion of correctly classified instances among all instances.  
It is the most basic and widely reported classification metric.

## 2. Formula
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:  
- \( TP \) = True Positives  
- \( TN \) = True Negatives  
- \( FP \) = False Positives  
- \( FN \) = False Negatives  

## 3. Interpretation
- Accuracy = 1 → Perfect classification.  
- Accuracy = 0 → All predictions are wrong.  
- Works best when classes are **balanced**.  

## 4. Advantages
- Simple to calculate and understand.  
- Useful as a baseline metric.  

## 5. Limitations
- Misleading for **imbalanced datasets** (e.g., 95% accuracy if predicting only the majority class).  
- Treats all errors equally, even when some are more costly.  

## 6. Use Cases
- Balanced classification tasks.  
- Quick model benchmarking.  


# Precision

## 1. What it is
**Precision** measures the proportion of positive predictions that are actually correct.  
It focuses on the reliability of positive predictions.

## 2. Formula
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

## 3. Interpretation
- High precision → Few false positives.  
- Low precision → Many false positives.  

## 4. Advantages
- Important when the **cost of a false positive** is high.  
- Intuitive for decision-making in domains like medicine, finance, or security.  

## 5. Limitations
- Ignores false negatives.  
- Alone, it does not reflect overall model performance.  

## 6. Use Cases
- Spam detection (how many predicted “spam” emails are actually spam).  
- Medical diagnostics (ensuring positive tests are correct).  


# Recall

## 1. What it is
**Recall** (also called **Sensitivity** or **True Positive Rate**) measures the proportion of actual positives that are correctly identified.  
It focuses on the model’s ability to **capture all relevant positive cases**.

## 2. Formula
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Where:  
- \( TP \) = True Positives  
- \( FN \) = False Negatives  

## 3. Interpretation
- High recall → Few false negatives (model rarely misses positives).  
- Low recall → Many false negatives (model misses positives often).  

## 4. Advantages
- Crucial when **missing a positive is costly**.  
- Complements precision for a fuller picture.  

## 5. Limitations
- Ignores false positives.  
- High recall alone may mean the model predicts too many positives (low precision).  

## 6. Use Cases
- Medical diagnostics (catching as many actual cases as possible).  
- Fraud detection (better to flag more, even at risk of some false alarms).  
- Information retrieval (ensuring relevant documents are not missed).  


# F1-Score

## 1. What it is
The **F1-Score** is the harmonic mean of Precision and Recall.  
It provides a single metric that balances both false positives and false negatives, especially useful when data is imbalanced.

## 2. Formula
$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Where:  
- Precision = \( \frac{TP}{TP + FP} \)  
- Recall = \( \frac{TP}{TP + FN} \)  

## 3. Interpretation
- F1 = 1 → Perfect Precision and Recall.  
- F1 = 0 → Worst possible performance.  
- More conservative than accuracy: both precision and recall must be high.  

## 4. Advantages
- Balances Precision and Recall in one metric.  
- Useful for imbalanced datasets.  
- Widely used in classification benchmarks.  

## 5. Limitations
- Ignores True Negatives (focuses only on positives).  
- Not always intuitive for non-technical audiences.  
- May not be ideal if Precision or Recall is clearly more important.  

## 6. Use Cases
- Text classification (spam filtering, sentiment analysis).  
- Medical testing (balancing between catching cases and reducing false alarms).  
- Fraud detection and anomaly detection.  


# Specificity (True Negative Rate)

## 1. What it is
**Specificity** measures the proportion of actual negatives that are correctly identified.  
It is the counterpart of Recall (Sensitivity), focusing on the **true negatives**.

## 2. Formula
$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

Where:  
- \( TN \) = True Negatives  
- \( FP \) = False Positives  

## 3. Interpretation
- High specificity → Few false positives (negatives rarely misclassified as positives).  
- Low specificity → Many false positives.  

## 4. Advantages
- Important when the **cost of a false positive** is high.  
- Complements Recall to give a full picture of performance.  

## 5. Limitations
- Ignores false negatives (focuses only on negatives).  
- Alone, it cannot describe model performance fully.  

## 6. Use Cases
- Medical testing (ensuring healthy people are not misdiagnosed as sick).  
- Security systems (avoiding false alarms).  
- Quality control (correctly rejecting defective vs. non-defective items).  


# Confusion Matrix

## 1. What it is
The **Confusion Matrix** is a tabular summary of classification results.  
It shows how many predictions fall into each category: true positives, true negatives, false positives, and false negatives.  
It is the foundation for many other classification metrics.

## 2. Structure
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)   | False Negative (FN)  |
| **Actual Negative** | False Positive (FP)  | True Negative (TN)   |

## 3. Interpretation
- TP → Correctly predicted positives.  
- TN → Correctly predicted negatives.  
- FP → Incorrectly predicted positives (Type I error).  
- FN → Missed positives (Type II error).  

## 4. Advantages
- Complete view of classification performance.  
- Basis for derived metrics (Accuracy, Precision, Recall, Specificity, F1).  
- Helps identify error types and class imbalance issues.  

## 5. Limitations
- Becomes less interpretable with many classes (multi-class confusion matrices can be large).  
- Not a single number metric; requires interpretation.  

## 6. Use Cases
- Diagnostic analysis of classification models.  
- Evaluating class imbalance impact.  
- Visualizing multi-class performance in ML experiments.  


# ROC Curve and AUC (Area Under the Curve)

## 1. What it is
The **Receiver Operating Characteristic (ROC) Curve** is a plot that shows the trade-off between **True Positive Rate (Recall)** and **False Positive Rate** across different decision thresholds.  
The **Area Under the ROC Curve (AUC)** summarizes this curve into a single value, measuring the model’s ability to distinguish between classes.

## 2. Formula
- False Positive Rate (FPR):  
$$
\text{FPR} = \frac{FP}{FP + TN}
$$

- True Positive Rate (TPR / Recall):  
$$
\text{TPR} = \frac{TP}{TP + FN}
$$

- **AUC**: The integral of the ROC curve:  
$$
\text{AUC} = \int_0^1 \text{TPR(FPR)} \, d(\text{FPR})
$$

## 3. Interpretation
- AUC = 1 → Perfect classifier.  
- AUC = 0.5 → Random guessing.  
- AUC < 0.5 → Worse than random (model is inverted).  

## 4. Advantages
- Threshold-independent (evaluates model across all thresholds).  
- Useful for comparing models.  
- Widely adopted in ML competitions and benchmarks.  

## 5. Limitations
- Can be overly optimistic with highly imbalanced datasets.  
- Does not reflect performance at a specific threshold.  
- ROC curve may look good even if Precision is poor.  

## 6. Use Cases
- Binary classification tasks.  
- Model comparison and selection.  
- Applications where ranking ability (not just classification at one threshold) matters.  


# Precision–Recall Curve and Average Precision (AP)

## 1. What it is
The **Precision–Recall (PR) Curve** plots Precision against Recall at different classification thresholds.  
It is especially informative for imbalanced datasets, where ROC curves may be misleading.  
The **Average Precision (AP)** summarizes the PR curve as the weighted average of Precision across Recall levels.

## 2. Formula
- Precision:  
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- Recall:  
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- Average Precision (AP):  
$$
AP = \sum_n (R_n - R_{n-1}) \cdot P_n
$$

Where:  
- \( P_n \) = Precision at threshold \( n \)  
- \( R_n \) = Recall at threshold \( n \)  

## 3. Interpretation
- High area under PR curve → Model has both high Precision and Recall.  
- AP = 1 → Perfect classification.  
- AP closer to baseline → Poor performance.  

## 4. Advantages
- More informative than ROC for imbalanced data.  
- Highlights trade-off between catching positives vs. avoiding false positives.  
- Standard metric in information retrieval and object detection.  

## 5. Limitations
- More sensitive to class imbalance than ROC.  
- Can be less intuitive for non-technical audiences.  

## 6. Use Cases
- Imbalanced datasets (fraud detection, rare disease diagnosis).  
- Information retrieval and ranking systems.  
- Computer vision tasks (object detection, image segmentation).  


# Log Loss (Cross-Entropy Loss)

## 1. What it is
**Log Loss**, also called **Cross-Entropy Loss**, measures the uncertainty of predictions by evaluating the difference between predicted probabilities and actual class labels.  
Unlike accuracy, which only checks if the predicted class is correct, Log Loss takes into account the **confidence** of the prediction.

## 2. Formula
For binary classification:
$$
\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^n \big[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \big]
$$

Where:  
- \( y_i \in \{0, 1\} \) = actual label  
- \( \hat{p}_i \) = predicted probability of class 1  

For multi-class:
$$
\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log(\hat{p}_{i,c})
$$

## 3. Interpretation
- Log Loss = 0 → Perfect predictions (probabilities match actual labels).  
- Lower values indicate better calibrated probability estimates.  
- Punishes both wrong predictions and overconfident predictions.  

## 4. Advantages
- Evaluates **probabilistic predictions**, not just hard classifications.  
- Proper scoring rule: encourages honest probability estimates.  
- Widely used in competitions and benchmarks.  

## 5. Limitations
- Sensitive to highly confident but wrong predictions (huge penalty).  
- Less intuitive to explain to non-technical audiences compared to accuracy.  

## 6. Use Cases
- Classification tasks where probabilities matter (finance, medicine).  
- Kaggle competitions and ML benchmarks.  
- Any application requiring **well-calibrated probability outputs**.  


# Hinge Loss

## 1. What it is
**Hinge Loss** is a classification loss function primarily used for training Support Vector Machines (SVMs).  
It focuses on maximizing the decision margin by penalizing not just wrong predictions, but also correct predictions that are **too close to the decision boundary**.

## 2. Formula
For binary classification with labels \( y_i \in \{-1, +1\} \):
$$
L(y_i, \hat{y}_i) = \max(0, 1 - y_i \cdot \hat{y}_i)
$$

Where:  
- \( y_i \) = true class label (\(-1\) or \(+1\))  
- \( \hat{y}_i \) = raw predicted score (before applying sign function)  

## 3. Interpretation
- Loss = 0 → Correct classification with a margin ≥ 1.  
- Positive loss → Either misclassified, or correctly classified but too close to boundary.  

## 4. Advantages
- Encourages **large margins**, improving generalization.  
- Convex and efficient for optimization.  
- Widely used in SVMs and linear classifiers.  

## 5. Limitations
- Requires labels in \(-1, +1\) form.  
- Not probabilistic — does not output well-calibrated probabilities.  
- Can be less interpretable compared to accuracy or F1.  

## 6. Use Cases
- Support Vector Machines (SVMs).  
- Linear classifiers with margin maximization.  
- Applications where separation and robustness are more important than probability estimates.  


# Matthews Correlation Coefficient (MCC)

## 1. What it is
The **Matthews Correlation Coefficient (MCC)** is a balanced measure of classification quality that considers all four elements of the confusion matrix: TP, TN, FP, and FN.  
It is essentially a correlation coefficient between actual and predicted classes, returning a value between -1 and +1.

## 2. Formula
$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

Where:  
- \( TP \) = True Positives  
- \( TN \) = True Negatives  
- \( FP \) = False Positives  
- \( FN \) = False Negatives  

## 3. Interpretation
- **MCC = +1** → Perfect classification.  
- **MCC = 0** → Random classification.  
- **MCC = -1** → Total disagreement between prediction and truth.  

## 4. Advantages
- Considers all confusion matrix components.  
- Robust to class imbalance.  
- Provides a single summary value (unlike Precision/Recall trade-offs).  

## 5. Limitations
- Less intuitive than Accuracy or F1 for non-technical audiences.  
- More complex to calculate manually.  

## 6. Use Cases
- Binary classification with imbalanced data.  
- Medical, fraud, or anomaly detection tasks where both positive and negative classes are important.  
- Model benchmarking in scientific research (often preferred over Accuracy or F1).  


# Cohen’s Kappa

## 1. What it is
**Cohen’s Kappa** measures the agreement between predicted and actual classifications, adjusted for the amount of agreement that could happen by chance.  
It is more robust than Accuracy, especially for imbalanced datasets.

## 2. Formula
$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Where:  
- \( p_o \) = observed agreement = \( \frac{TP + TN}{TP + TN + FP + FN} \)  
- \( p_e \) = expected agreement by chance, calculated from marginal probabilities.  

## 3. Interpretation
- \( \kappa = 1 \) → Perfect agreement.  
- \( \kappa = 0 \) → Agreement is no better than chance.  
- \( \kappa < 0 \) → Worse than chance (systematic disagreement).  

## 4. Advantages
- Adjusts for chance agreement.  
- Useful for imbalanced datasets.  
- Widely used in research, especially inter-rater reliability.  

## 5. Limitations
- Can be less intuitive than Accuracy or F1.  
- Sensitive to class distribution (skewed prevalence can lower kappa).  
- Does not provide insight into *type* of errors (needs confusion matrix context).  

## 6. Use Cases
- Classification evaluation with imbalanced classes.  
- Inter-rater agreement studies (medicine, linguistics, psychology).  
- Multi-class classification when balanced evaluation is needed.  


# Balanced Accuracy

## 1. What it is
**Balanced Accuracy** is the average of Recall (Sensitivity) and Specificity.  
It was designed to handle imbalanced datasets better than standard Accuracy, by giving equal weight to both classes.

## 2. Formula
$$
\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}
$$

Where:  
- Sensitivity (Recall) = \( \frac{TP}{TP + FN} \)  
- Specificity = \( \frac{TN}{TN + FP} \)  

## 3. Interpretation
- Balanced Accuracy = 1 → Perfect classification.  
- Balanced Accuracy = 0.5 → Random guessing (binary case).  
- More fair when classes are imbalanced.  

## 4. Advantages
- Corrects bias of plain Accuracy on imbalanced datasets.  
- Easy to compute and interpret.  
- Directly incorporates both positive and negative class performance.  

## 5. Limitations
- Still a single-number summary, may hide trade-offs between Precision and Recall.  
- Less common than Accuracy or F1 in business reports, so harder to communicate.  

## 6. Use Cases
- Imbalanced binary classification (fraud detection, medical diagnostics).  
- Benchmarking models when both minority and majority classes matter.  
- Situations where both sensitivity and specificity are equally important.  


# Fβ-Score

## 1. What it is
The **Fβ-Score** is a generalization of the F1-Score that allows you to put more weight on either Precision or Recall.  
The parameter \( \beta \) controls the balance:  
- \( \beta > 1 \) → Recall is more important.  
- \( \beta < 1 \) → Precision is more important.  
- \( \beta = 1 \) → Equal weight (F1-Score).  

## 2. Formula
$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}
$$

Where:  
- Precision = \( \frac{TP}{TP + FP} \)  
- Recall = \( \frac{TP}{TP + FN} \)  

## 3. Interpretation
- \( F_\beta = 1 \) → Perfect Precision and Recall.  
- Adjusts the trade-off depending on problem needs.  

## 4. Advantages
- Flexible: allows prioritizing Recall or Precision.  
- Useful when the cost of false negatives ≠ false positives.  
- Keeps harmonic mean formulation for balance.  

## 5. Limitations
- Choice of \( \beta \) is problem-dependent (must be decided carefully).  
- Less interpretable for non-technical audiences than F1.  

## 6. Use Cases
- Medical testing: use \( \beta > 1 \) (Recall more important than Precision).  
- Spam detection: use \( \beta < 1 \) (Precision more important to avoid misclassifying real emails).  
- Any imbalanced classification task with asymmetric error costs.  


# Brier Score

## 1. What it is
The **Brier Score** measures the mean squared difference between predicted probabilities and the actual class labels (0 or 1).  
It evaluates both the **calibration** and **accuracy** of probabilistic predictions.

## 2. Formula
For binary classification:
$$
\text{Brier Score} = \frac{1}{n} \sum_{i=1}^n \left( \hat{p}_i - y_i \right)^2
$$

Where:  
- \( y_i \in \{0, 1\} \) = actual label  
- \( \hat{p}_i \) = predicted probability for the positive class  

For multi-class classification:
$$
\text{Brier Score} = \frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C \left( \hat{p}_{i,c} - y_{i,c} \right)^2
$$

## 3. Interpretation
- Brier Score = 0 → Perfect probabilistic predictions.  
- Lower values = better performance.  
- Range: [0, 1] for binary tasks.  

## 4. Advantages
- Evaluates probability estimates directly (not just hard labels).  
- Proper scoring rule: encourages well-calibrated probabilities.  
- Works for both binary and multi-class tasks.  

## 5. Limitations
- Sensitive to class imbalance.  
- Less intuitive than Accuracy or F1.  
- Can be hard to explain to non-technical audiences.  

## 6. Use Cases
- Weather forecasting (e.g., “70% chance of rain”).  
- Credit scoring and risk prediction.  
- Any classification task where probability calibration matters.  


# Top-k Accuracy

## 1. What it is
**Top-k Accuracy** evaluates whether the true class is among the model’s top \( k \) predicted classes.  
It is commonly used when models output probability distributions over many classes (e.g., deep learning for image classification).

## 2. Formula
For each sample \( i \):  
- The prediction is correct if the true label \( y_i \) is in the top-\( k \) predicted probabilities.  

Formally:
$$
\text{Top-}k \, \text{Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in \text{Top-}k(\hat{p}_i) \}
$$

Where:  
- \( n \) = number of samples  
- \( \hat{p}_i \) = predicted probability distribution for sample \( i \)  
- \( \mathbf{1}\{\cdot\} \) = indicator function (1 if true, 0 otherwise)  

## 3. Interpretation
- Top-1 Accuracy = standard Accuracy.  
- Top-5 Accuracy = checks if the true label is in the top 5 predictions.  
- Higher \( k \) → more lenient evaluation.  

## 4. Advantages
- Useful in problems with many classes.  
- More realistic in applications where multiple suggestions are acceptable (e.g., recommendation systems).  

## 5. Limitations
- Does not evaluate probability calibration, only ranking.  
- Larger \( k \) values may inflate accuracy, giving false confidence.  

## 6. Use Cases
- Image classification (e.g., ImageNet benchmarks report Top-1 and Top-5 Accuracy).  
- Recommendation systems (whether the correct item is in the suggested list).  
- NLP tasks with large vocabularies.  


# Hamming Loss

## 1. What it is
**Hamming Loss** measures the fraction of incorrect labels (both false positives and false negatives) in multi-label classification problems.  
It evaluates how often predicted labels differ from the true set of labels.

## 2. Formula
For \( n \) samples and \( L \) labels:
$$
\text{Hamming Loss} = \frac{1}{n \cdot L} \sum_{i=1}^n \sum_{j=1}^L \mathbf{1}\{ y_{i,j} \neq \hat{y}_{i,j} \}
$$

Where:  
- \( y_{i,j} \in \{0,1\} \) = true label for class \( j \) in sample \( i \)  
- \( \hat{y}_{i,j} \in \{0,1\} \) = predicted label  
- \( \mathbf{1}\{\cdot\} \) = indicator function  

## 3. Interpretation
- Hamming Loss = 0 → Perfect classification.  
- Higher values → More label mismatches.  
- Range: [0,1].  

## 4. Advantages
- Suitable for multi-label tasks (where multiple labels may apply to one instance).  
- Penalizes both false positives and false negatives equally.  

## 5. Limitations
- Does not account for label importance (all labels treated equally).  
- Can be misleading if some labels are very rare.  

## 6. Use Cases
- Text classification with multiple topics per document.  
- Image tagging (an image may belong to multiple categories).  
- Medical diagnosis (patients may have multiple conditions simultaneously).  


# Jaccard Index (Intersection over Union, IoU)

## 1. What it is
The **Jaccard Index**, also known as **Intersection over Union (IoU)**, measures the similarity between the predicted set of labels and the actual set of labels.  
It is widely used for **multi-label classification** and **image segmentation**.

## 2. Formula
For two sets \( A \) (predicted) and \( B \) (actual):
$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

For binary classification:
$$
J = \frac{TP}{TP + FP + FN}
$$

## 3. Interpretation
- Jaccard = 1 → Perfect agreement.  
- Jaccard = 0 → No overlap between prediction and truth.  
- Range: [0,1].  

## 4. Advantages
- Captures both false positives and false negatives.  
- Suitable for multi-label and structured prediction tasks.  
- Intuitive measure of overlap between prediction and truth.  

## 5. Limitations
- Ignores True Negatives.  
- Can be harsh for highly imbalanced labels (small overlap reduces score drastically).  

## 6. Use Cases
- Multi-label classification (documents with multiple topics).  
- Image segmentation (overlap between predicted and ground-truth masks).  
- Object detection (measuring bounding box accuracy).  


# Geometric Mean (G-Mean)

## 1. What it is
The **Geometric Mean (G-Mean)** evaluates the balance between **Sensitivity (Recall)** and **Specificity**.  
It is especially useful for imbalanced datasets, ensuring that performance on both positive and negative classes is considered.

## 2. Formula
$$
\text{G-Mean} = \sqrt{\text{Sensitivity} \cdot \text{Specificity}}
$$

Where:  
- Sensitivity (Recall) = \( \frac{TP}{TP + FN} \)  
- Specificity = \( \frac{TN}{TN + FP} \)  

## 3. Interpretation
- G-Mean = 1 → Perfect classification (both classes predicted well).  
- Low G-Mean → Poor performance on at least one class.  

## 4. Advantages
- Handles imbalanced datasets better than plain Accuracy.  
- Considers both false positives and false negatives.  
- Encourages balanced performance across classes.  

## 5. Limitations
- Limited to binary classification (extensions exist for multi-class but less common).  
- Less intuitive than Accuracy or F1.  

## 6. Use Cases
- Medical diagnosis (balancing detection of disease vs. healthy cases).  
- Fraud detection (catching fraud while avoiding false alarms).  
- Any highly imbalanced classification problem.  


# Macro, Micro, and Weighted Averages

## 1. What it is
In **multi-class or multi-label classification**, metrics such as Precision, Recall, and F1 can be aggregated in different ways:  
- **Macro Average**: Calculates the metric independently for each class, then takes the unweighted mean. Treats all classes equally.  
- **Micro Average**: Aggregates contributions of all classes (summing TP, FP, FN) before calculating the metric. Favors performance on majority classes.  
- **Weighted Average**: Like Macro Average, but each class’s metric is weighted by its support (number of true instances).  

## 2. Formulas
- Macro Precision:
$$
\text{Precision}_{macro} = \frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}
$$

- Micro Precision (same for Recall and F1):
$$
\text{Precision}_{micro} = \frac{\sum_{c=1}^C TP_c}{\sum_{c=1}^C (TP_c + FP_c)}
$$

- Weighted Precision:
$$
\text{Precision}_{weighted} = \frac{\sum_{c=1}^C n_c \cdot \text{Precision}_c}{\sum_{c=1}^C n_c}
$$

Where:  
- \( C \) = number of classes  
- \( n_c \) = number of samples in class \( c \)  

## 3. Interpretation
- **Macro**: Good for imbalanced data if you want all classes to matter equally.  
- **Micro**: Good if overall sample-level performance matters most.  
- **Weighted**: Balances per-class performance but reflects dataset imbalance.  

## 4. Advantages
- Allow adapting metrics to dataset characteristics.  
- Provide flexibility in evaluating multi-class problems.  

## 5. Limitations
- Choice of averaging method can change results drastically.  
- Macro average may overemphasize rare classes.  
- Micro average may hide poor performance on minority classes.  

## 6. Use Cases
- Multi-class text classification, image recognition.  
- Reporting classification results in imbalanced datasets.  
- Benchmarking models fairly across classes.  


# AUCPR (Area Under the Precision–Recall Curve)

## 1. What it is
The **Area Under the Precision–Recall Curve (AUCPR)** summarizes the trade-off between Precision and Recall across different thresholds.  
It is similar to ROC-AUC but more informative when dealing with **imbalanced datasets**, where the positive class is rare.

## 2. Formula
The AUCPR is computed as the area under the PR curve:
$$
\text{AUCPR} = \int_0^1 \text{Precision(Recall)} \, d(\text{Recall})
$$

In practice, it is often estimated numerically from Precision–Recall pairs at different thresholds.

## 3. Interpretation
- AUCPR = 1 → Perfect classifier.  
- AUCPR close to the baseline (positive class prevalence) → Poor performance.  
- Always ≤ 1.  

## 4. Advantages
- More informative than ROC-AUC for imbalanced data.  
- Focuses on positive class performance (important in rare-event detection).  
- Widely used in information retrieval, anomaly detection, and medical AI.  

## 5. Limitations
- Less intuitive than ROC-AUC for general audiences.  
- Can vary significantly with class prevalence.  
- Sensitive to small changes in Recall when positives are rare.  

## 6. Use Cases
- Imbalanced binary classification (fraud detection, medical diagnosis).  
- Information retrieval tasks.  
- Object detection and segmentation (evaluating rare object categories).  


# Focal Loss

## 1. What it is
The **Focal Loss** is a modified version of Cross-Entropy Loss that focuses training on **hard-to-classify examples** while reducing the loss contribution from easy, well-classified examples.  
It was introduced for object detection tasks (e.g., RetinaNet) to deal with severe class imbalance.

## 2. Formula
For binary classification:
$$
FL(p_t) = - \alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)
$$

Where:  
- \( p_t \) = predicted probability for the true class  
- \( \alpha \) = balancing factor between classes (0 < α < 1)  
- \( \gamma \) = focusing parameter (γ ≥ 0)  

## 3. Interpretation
- When \( p_t \) is high (easy example), the loss is down-weighted.  
- When \( p_t \) is low (hard example), the loss is amplified.  
- \( \gamma \) controls how much to focus on hard examples.  

## 4. Advantages
- Effective for imbalanced datasets.  
- Improves training stability in object detection and segmentation tasks.  
- Generalizes Cross-Entropy Loss (when γ = 0, it reduces to standard CE).  

## 5. Limitations
- Requires tuning of hyperparameters (α, γ).  
- Less intuitive than Accuracy, F1, or ROC-AUC.  
- More computationally intensive than plain CE.  

## 6. Use Cases
- Object detection (e.g., RetinaNet).  
- Imbalanced medical datasets (rare disease detection).  
- Any classification problem with severe class imbalance.  


# Cumulative Gain and Lift Curve

## 1. What it is
**Cumulative Gain** and **Lift Curves** are evaluation tools for classification models that output probabilities or scores.  
They help visualize how well the model captures positive instances compared to a random model.  

- **Cumulative Gain Curve**: Shows the percentage of actual positives captured by selecting the top \( x\% \) of predicted instances.  
- **Lift Curve**: Measures how many times better the model is at identifying positives compared to random selection.

## 2. Formula
- Cumulative Gain at \( k\% \):
$$
\text{Gain}(k) = \frac{\text{Number of true positives in top } k\%}{\text{Total number of positives}}
$$

- Lift at \( k\% \):
$$
\text{Lift}(k) = \frac{\text{Gain}(k)}{k}
$$

## 3. Interpretation
- Gain Curve above the diagonal → Model is better than random.  
- Lift > 1 → Model is performing better than random selection.  
- Higher Lift values indicate stronger model discrimination.  

## 4. Advantages
- Very intuitive for business applications.  
- Helps in resource allocation (e.g., targeting top customers).  
- Useful for probability ranking tasks.  

## 5. Limitations
- Not as common in pure ML research (more used in applied domains).  
- Sensitive to class imbalance.  
- Does not directly measure misclassification costs.  

## 6. Use Cases
- Marketing campaigns (targeting customers most likely to respond).  
- Credit scoring and risk models (identifying high-risk customers).  
- Fraud detection (ranking suspicious transactions).  


# Expected Calibration Error (ECE)

## 1. What it is
The **Expected Calibration Error (ECE)** measures how well predicted probabilities reflect the true likelihood of outcomes.  
A well-calibrated model means that if it predicts something with 70% confidence, it should be correct about 70% of the time.

## 2. Formula
ECE is computed by binning predictions into \( M \) intervals (e.g., 0–0.1, 0.1–0.2, …), then comparing accuracy within each bin to the average predicted probability:

$$
ECE = \sum_{m=1}^M \frac{|B_m|}{n} \; \big| \text{acc}(B_m) - \text{conf}(B_m) \big|
$$

Where:  
- \( B_m \) = set of samples in bin \( m \)  
- \( n \) = total number of samples  
- \( \text{acc}(B_m) \) = accuracy in bin \( m \)  
- \( \text{conf}(B_m) \) = average predicted confidence in bin \( m \)  

## 3. Interpretation
- ECE = 0 → Perfect calibration.  
- Higher values → Worse calibration (predicted probabilities don’t match actual outcomes).  

## 4. Advantages
- Captures the difference between predicted confidence and actual correctness.  
- Useful for tasks where well-calibrated probabilities are critical.  
- Easy to compute and visualize with reliability diagrams.  

## 5. Limitations
- Depends on binning strategy (choice of \( M \)).  
- Does not capture variance within bins.  
- Not as widely reported as accuracy or AUC in standard ML benchmarks.  

## 6. Use Cases
- Medical AI (probability estimates must be trustworthy).  
- Finance and risk prediction.  
- Any application where predicted probabilities are directly used for decision-making.  


# Kolmogorov–Smirnov (KS) Statistic

## 1. What it is
The **Kolmogorov–Smirnov (KS) Statistic** measures the maximum separation between the cumulative distribution of predicted probabilities for the positive and negative classes.  
It indicates how well a model distinguishes between the two classes.

## 2. Formula
$$
KS = \max_x \big| F_{1}(x) - F_{0}(x) \big|
$$

Where:  
- \( F_{1}(x) \) = cumulative distribution of predicted probabilities for positives  
- \( F_{0}(x) \) = cumulative distribution of predicted probabilities for negatives  

## 3. Interpretation
- KS = 0 → No separation (model cannot distinguish classes).  
- KS = 1 → Perfect separation.  
- Higher KS values → Better discrimination.  

## 4. Advantages
- Simple and interpretable.  
- Focuses on ranking and separation ability.  
- Standard in finance, credit scoring, and risk modeling.  

## 5. Limitations
- Less common outside risk/finance industries.  
- Only applies to binary classification.  
- Does not directly reflect probability calibration.  

## 6. Use Cases
- Credit scoring (distinguishing good vs. bad loans).  
- Risk models in banking and insurance.  
- Fraud detection (separating fraudulent vs. legitimate transactions).  


# Gini Coefficient

## 1. What it is
The **Gini Coefficient** is a measure of inequality that has been adapted as a classification metric.  
In classification, it measures how well the model separates the positive and negative classes and is directly related to the ROC-AUC.

## 2. Formula
$$
\text{Gini} = 2 \cdot \text{AUC} - 1
$$

Where:  
- AUC = Area Under the ROC Curve  

## 3. Interpretation
- Gini = 1 → Perfect classification.  
- Gini = 0 → Model performs no better than random guessing.  
- Gini < 0 → Model is worse than random (inverted predictions).  

## 4. Advantages
- Easy to compute from ROC-AUC.  
- Widely used in finance and risk modeling.  
- Scales between -1 and 1 for interpretability.  

## 5. Limitations
- Provides no extra information beyond ROC-AUC.  
- Less intuitive than Precision/Recall for many ML practitioners.  
- Not as widely used in general ML research.  

## 6. Use Cases
- Credit scoring (standard evaluation metric in banking).  
- Marketing and customer churn models.  
- Any binary classification problem where ROC-AUC is used.  


# Youden’s J Statistic (Informedness)

## 1. What it is
**Youden’s J Statistic**, also known as **Informedness**, is a single-number summary of a binary classifier’s performance at a given threshold.  
It combines Sensitivity (Recall) and Specificity, measuring the probability of an informed decision versus a random guess.

## 2. Formula
$$
J = \text{Sensitivity} + \text{Specificity} - 1
$$

Where:  
- Sensitivity = \( \frac{TP}{TP + FN} \)  
- Specificity = \( \frac{TN}{TN + FP} \)  

## 3. Interpretation
- J = 1 → Perfect classifier.  
- J = 0 → No better than random guessing.  
- J < 0 → Systematically wrong classification.  

## 4. Advantages
- Simple and interpretable.  
- Balances both positive and negative class performance.  
- Useful for selecting an optimal threshold in binary classification.  

## 5. Limitations
- Threshold-dependent.  
- Less informative for highly imbalanced datasets.  
- Not widely used outside medical and diagnostic testing.  

## 6. Use Cases
- Medical diagnostics (choosing optimal threshold for tests).  
- Binary classification problems with balanced importance of false positives and false negatives.  
- Quick comparison of classifier thresholds.  


# False Positive Rate (FPR)

## 1. What it is
The **False Positive Rate (FPR)** measures the proportion of actual negatives that are incorrectly classified as positives.  
It is also known as the **Fall-out**.

## 2. Formula
$$
\text{FPR} = \frac{FP}{FP + TN}
$$

Where:  
- \( FP \) = False Positives  
- \( TN \) = True Negatives  

## 3. Interpretation
- FPR = 0 → No negatives misclassified as positives.  
- FPR = 1 → All negatives misclassified as positives.  

## 4. Advantages
- Directly captures Type I errors.  
- Important when false positives are costly.  

## 5. Limitations
- Ignores false negatives.  
- Alone, does not provide full classifier performance.  

## 6. Use Cases
- Spam detection (minimizing false alarms on genuine emails).  
- Security systems (avoiding false alarms).  
- Medical screening tests.  


# False Negative Rate (FNR)

## 1. What it is
The **False Negative Rate (FNR)** measures the proportion of actual positives that are incorrectly classified as negatives.  
It is also known as the **Miss Rate**.

## 2. Formula
$$
\text{FNR} = \frac{FN}{FN + TP}
$$

Where:  
- \( FN \) = False Negatives  
- \( TP \) = True Positives  

## 3. Interpretation
- FNR = 0 → No positives misclassified as negatives.  
- FNR = 1 → All positives missed.  

## 4. Advantages
- Directly captures Type II errors.  
- Crucial when missing positives is costly.  

## 5. Limitations
- Ignores false positives.  
- Alone, does not capture overall classifier performance.  

## 6. Use Cases
- Medical diagnosis (avoiding missed cases).  
- Fraud detection (ensuring fraudulent transactions are not overlooked).  
- Rare-event detection tasks.  


# Balanced Error Rate (BER)

## 1. What it is
The **Balanced Error Rate (BER)** is the average of the False Positive Rate (FPR) and the False Negative Rate (FNR).  
It balances errors across both classes, making it useful for **imbalanced datasets**.

## 2. Formula
$$
\text{BER} = \frac{1}{2} \left( \frac{FP}{FP + TN} + \frac{FN}{FN + TP} \right)
$$

Or equivalently:
$$
\text{BER} = 1 - \frac{\text{Sensitivity} + \text{Specificity}}{2}
$$

Where:  
- Sensitivity = Recall = \( \frac{TP}{TP + FN} \)  
- Specificity = \( \frac{TN}{TN + FP} \)  

## 3. Interpretation
- BER = 0 → Perfect classification.  
- BER = 0.5 → Random guessing (binary case).  
- Lower BER → Better classifier.  

## 4. Advantages
- Corrects for class imbalance (unlike raw Accuracy).  
- Directly relates to Sensitivity and Specificity.  
- Easy to compute.  

## 5. Limitations
- Only applies to binary classification (multi-class extensions exist but less common).  
- Less widely reported compared to Accuracy or F1.  

## 6. Use Cases
- Imbalanced binary classification tasks.  
- Medical testing (equal focus on sensitivity and specificity).  
- Fraud detection.  


# Macro-AUC, Micro-AUC, and Weighted-AUC

## 1. What it is
When extending ROC-AUC to multi-class classification, different averaging strategies are used:
- **Macro-AUC**: Compute ROC-AUC for each class (one-vs-rest), then take the unweighted mean. Treats all classes equally.  
- **Micro-AUC**: Aggregate predictions across all classes, then compute a single ROC-AUC. Favors majority classes.  
- **Weighted-AUC**: Like Macro, but each class AUC is weighted by its support (number of true instances).  

## 2. Interpretation
- Macro-AUC → Best when all classes are equally important.  
- Micro-AUC → Best when overall performance matters most.  
- Weighted-AUC → Balances importance by class distribution.  

## 3. Advantages
- Extend ROC-AUC naturally to multi-class problems.  
- Flexible depending on class distribution and goals.  

## 4. Limitations
- Different averaging strategies may lead to different conclusions.  
- Micro-AUC can hide poor performance on minority classes.  

## 5. Use Cases
- Multi-class classification tasks (image recognition, text categorization).  
- Imbalanced datasets where class-importance differs.  


# Class-Weighted Metrics

## 1. What it is
In imbalanced classification, **Class-Weighted Metrics** adjust standard metrics (e.g., Precision, Recall, F1) by assigning weights to each class proportional to their importance or prevalence.  
This prevents majority classes from dominating the evaluation.

## 2. Formula
For F1-Score:
$$
F1_{weighted} = \frac{\sum_{c=1}^C n_c \cdot F1_c}{\sum_{c=1}^C n_c}
$$

Where:  
- \( C \) = number of classes  
- \( n_c \) = number of true instances in class \( c \)  
- \( F1_c \) = class-specific F1 score  

## 3. Interpretation
- Weighted metrics give more influence to frequent classes.  
- Useful when reporting a single score across imbalanced datasets.  

## 4. Advantages
- More fair than plain averages when dataset is skewed.  
- Keeps minority class performance visible.  

## 5. Limitations
- Still dominated by majority class if imbalance is extreme.  
- Choice of weights (frequency vs. custom) may change results.  

## 6. Use Cases
- Any imbalanced dataset.  
- Credit scoring, fraud detection, churn prediction.  


# Fowlkes–Mallows Index (FMI)

## 1. What it is
The **Fowlkes–Mallows Index (FMI)** measures the geometric mean of Precision and Recall.  
Although more common in clustering evaluation, it can also be applied to binary or multi-class classification.

## 2. Formula
$$
\text{FMI} = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}
$$

Where:  
- \( \frac{TP}{TP + FP} \) = Precision  
- \( \frac{TP}{TP + FN} \) = Recall  

## 3. Interpretation
- FMI = 1 → Perfect Precision and Recall.  
- FMI close to 0 → Poor classification.  

## 4. Advantages
- Balances Precision and Recall like F1, but with geometric mean.  
- Symmetric and interpretable.  

## 5. Limitations
- Less common than F1 in ML research.  
- Same drawbacks as Precision/Recall trade-off.  

## 6. Use Cases
- Multi-class and clustering tasks.  
- Alternative to F1-Score in imbalanced data scenarios.  


# Fleiss’ Kappa (Multi-Rater Agreement)

## 1. What it is
**Fleiss’ Kappa** is an extension of Cohen’s Kappa to measure agreement among more than two raters (e.g., multiple annotators labeling data).  
It adjusts for agreement expected by chance.

## 2. Formula
$$
\kappa = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}
$$

Where:  
- \( \bar{P} \) = observed average agreement  
- \( \bar{P}_e \) = expected agreement by chance  

## 3. Interpretation
- κ = 1 → Perfect agreement.  
- κ = 0 → No better than random.  
- κ < 0 → Worse than random.  

## 4. Advantages
- Useful for data labeling quality assessment.  
- Extends Cohen’s Kappa to multi-rater settings.  

## 5. Limitations
- Sensitive to class imbalance.  
- Interpretation depends on context.  

## 6. Use Cases
- Annotator agreement in NLP datasets.  
- Medical imaging (multiple radiologists labeling scans).  
- Crowdsourced labeling tasks.  


# Partial AUC (pAUC)

## 1. What it is
The **Partial Area Under the ROC Curve (pAUC)** restricts AUC calculation to a specific range of False Positive Rates (FPR).  
This is especially relevant in medical and safety-critical applications, where only part of the ROC curve matters.

## 2. Formula
$$
\text{pAUC} = \int_{FPR_{min}}^{FPR_{max}} TPR(FPR) \, dFPR
$$

Where:  
- Integration is limited to a subset of the ROC curve.  

## 3. Interpretation
- pAUC focuses only on the region of practical interest.  
- Higher pAUC → better classifier in the chosen range.  

## 4. Advantages
- Tailored to domain-specific thresholds.  
- Useful when low false positive rates are critical.  

## 5. Limitations
- Ignores parts of ROC curve outside the chosen region.  
- Less comparable across different studies.  

## 6. Use Cases
- Medical tests (e.g., cancer screening, where FPR must be very low).  
- Security and fraud detection.  


# Multi-class MCC (Generalization)

## 1. What it is
The **Multi-class Matthews Correlation Coefficient (MCC)** extends MCC to more than two classes.  
It still returns a value between -1 and +1.

## 2. Formula
For confusion matrix \( C \) with \( K \) classes:
$$
\text{MCC} = \frac{\sum_{k}\sum_{l}\sum_{m} C_{kk} C_{lm} - C_{kl} C_{mk}}
{\sqrt{(\sum_{k} t_k)(\sum_{k} p_k)(\sum_{k} n_k)(\sum_{k} m_k)}}
$$

Where:  
- \( t_k \) = number of true samples of class \( k \)  
- \( p_k \) = number of predicted samples of class \( k \)  
- \( n_k, m_k \) = derived counts across classes  

## 3. Interpretation
- MCC = 1 → Perfect classification.  
- MCC = 0 → Random performance.  
- MCC = -1 → Inverted predictions.  

## 4. Advantages
- Handles multi-class imbalanced problems fairly.  
- Considers entire confusion matrix.  

## 5. Limitations
- Formula is more complex.  
- Less interpretable than Accuracy or F1.  

## 6. Use Cases
- Multi-class classification in NLP, CV.  
- Medical diagnostics with multiple disease categories.  


# Label Ranking Average Precision (LRAP)

## 1. What it is
The **Label Ranking Average Precision (LRAP)** evaluates ranking quality in **multi-label classification**.  
It measures how well the model ranks true labels above false labels for each sample.

## 2. Formula
For each sample \( i \):
$$
LRAP = \frac{1}{n} \sum_{i=1}^n \frac{1}{|Y_i|} \sum_{y \in Y_i} \frac{|\{ y' \in Y_i : \hat{p}_{i,y'} \geq \hat{p}_{i,y} \}|}{\text{rank of } y}
$$

Where:  
- \( Y_i \) = set of true labels for sample \( i \)  
- \( \hat{p}_{i,y} \) = predicted probability for label \( y \)  

## 3. Interpretation
- LRAP = 1 → Perfect ranking (all true labels ranked above false ones).  
- Closer to 0 → Poor label ranking.  

## 4. Advantages
- Specifically designed for multi-label ranking.  
- More informative than plain accuracy for multi-label data.  

## 5. Limitations
- Harder to interpret compared to Precision/Recall.  
- Less common outside research and benchmarking.  

## 6. Use Cases
- Multi-label classification (text tagging, music genre classification).  
- Recommendation systems.  
- NLP tasks with overlapping categories.  


# Ranking & Information Retrieval Metrics

Ranking and information retrieval metrics evaluate how well a model orders or retrieves relevant items, documents, or recommendations.  
They are crucial in **search engines, recommender systems, and information retrieval tasks**, where the position of results matters as much as their correctness.


# Precision@k

## 1. What it is
**Precision@k** measures the proportion of relevant items among the top-\( k \) retrieved or recommended results.  
It focuses on how many of the top-\( k \) outputs are correct.

## 2. Formula
$$
\text{Precision@}k = \frac{\text{Number of relevant items in top-}k}{k}
$$

## 3. Interpretation
- Precision@5 = 0.6 → 3 out of the top 5 results are relevant.  
- Higher Precision@k means the model ranks relevant items higher.  

## 4. Advantages
- Simple and intuitive.  
- Focuses on user experience (top-ranked items matter most).  
- Flexible: can evaluate different cutoffs (k = 5, 10, 20, etc.).  

## 5. Limitations
- Ignores results beyond rank \( k \).  
- Does not account for the order within top-\( k \).  
- Sensitive to choice of \( k \).  

## 6. Use Cases
- Search engines (relevant documents in top results).  
- Recommender systems (movies, products, or songs).  
- Information retrieval tasks in NLP.  


# Recall@k

## 1. What it is
**Recall@k** measures the proportion of relevant items retrieved among all possible relevant items, considering only the top-\( k \) results.  
It captures how well the model retrieves the complete set of relevant results.

## 2. Formula
$$
\text{Recall@}k = \frac{\text{Number of relevant items in top-}k}{\text{Total number of relevant items}}
$$

## 3. Interpretation
- Recall@10 = 0.8 → 80% of all relevant items are found within the top 10.  
- Higher Recall@k means better coverage of relevant items.  

## 4. Advantages
- Focuses on completeness of retrieval.  
- Useful when finding **all relevant results** matters.  

## 5. Limitations
- Ignores ranking within top-\( k \).  
- May overvalue large result sets (retrieving many items boosts recall but lowers precision).  

## 6. Use Cases
- Legal/medical search (must retrieve all relevant documents).  
- Recommendation systems where recall of options matters (music/movie suggestions).  


# Mean Average Precision (MAP)

## 1. What it is
**Mean Average Precision (MAP)** measures ranking quality by averaging Precision across recall levels for each query, then averaging over all queries.  
It balances both precision and recall across the entire ranking.

## 2. Formula
For a single query:
$$
AP = \frac{1}{R} \sum_{k=1}^n P(k) \cdot rel(k)
$$

Where:  
- \( R \) = number of relevant items  
- \( P(k) \) = Precision@k  
- \( rel(k) = 1 \) if the item at rank \( k \) is relevant, else 0  

MAP across all queries:
$$
MAP = \frac{1}{Q} \sum_{q=1}^Q AP(q)
$$

## 3. Interpretation
- MAP = 1 → Perfect ranking (all relevant items ranked at the top).  
- Higher MAP → Better ranking consistency across queries.  

## 4. Advantages
- Considers order of relevant results.  
- Robust for multi-query evaluation.  

## 5. Limitations
- Computationally more intensive than simple Precision/Recall.  
- May not be intuitive for non-technical users.  

## 6. Use Cases
- Search engine evaluation (Google, Bing).  
- Multi-query retrieval tasks (question answering).  
- Recommendation system benchmarking.  


# Normalized Discounted Cumulative Gain (NDCG)

## 1. What it is
**Normalized Discounted Cumulative Gain (NDCG)** evaluates ranking quality by assigning higher weight to relevant results at higher ranks, with a logarithmic discount as rank increases.  
It is widely used in information retrieval.

## 2. Formula
Discounted Cumulative Gain (DCG):
$$
DCG@k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

Normalized DCG:
$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

Where:  
- \( rel_i \) = relevance score of item at position \( i \)  
- \( IDCG@k \) = DCG of the ideal (perfect) ranking  

## 3. Interpretation
- NDCG = 1 → Perfect ranking.  
- Values close to 0 → Poor ranking.  
- Considers both order and graded relevance.  

## 4. Advantages
- Accounts for ranking order and relevance scores.  
- Discounts lower-ranked results (realistic for user behavior).  
- Handles graded relevance, not just binary relevance.  

## 5. Limitations
- Requires relevance scores (not always available).  
- Less intuitive than Precision@k.  

## 6. Use Cases
- Search engines (ranking web pages by relevance).  
- Recommendation systems with ratings.  
- NLP information retrieval tasks.  


# Mean Reciprocal Rank (MRR)

## 1. What it is
**Mean Reciprocal Rank (MRR)** evaluates how highly the first relevant item is ranked in the results.  
It is the average of reciprocal ranks across queries.

## 2. Formula
For a single query:
$$
RR = \frac{1}{rank_{first\ relevant}}
$$

For multiple queries:
$$
MRR = \frac{1}{Q} \sum_{q=1}^Q RR_q
$$

## 3. Interpretation
- MRR = 1 → Every query’s first relevant result is at rank 1.  
- Higher MRR → Relevant items appear earlier in the ranking.  

## 4. Advantages
- Very simple and interpretable.  
- Focuses on user experience (first relevant hit matters most).  

## 5. Limitations
- Ignores results beyond the first relevant item.  
- Not suitable when multiple relevant items matter.  

## 6. Use Cases
- Question answering systems (is the correct answer ranked first?).  
- Search engines (time to first relevant document).  
- Conversational AI retrieval systems.  


# Hit Ratio (HR@k)

## 1. What it is
**Hit Ratio at k (HR@k)** checks whether at least one relevant item appears in the top-\( k \) results.  
Unlike Precision@k, it only cares about a "hit" rather than the number of relevant items.

## 2. Formula
$$
HR@k = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ \text{any relevant item in top-}k \}
$$

## 3. Interpretation
- HR@10 = 0.9 → In 90% of cases, at least one relevant item is in the top 10.  

## 4. Advantages
- Very intuitive.  
- Useful when just one correct suggestion is enough.  

## 5. Limitations
- Ignores number of relevant items.  
- Less strict than Precision/Recall.  

## 6. Use Cases
- Recommender systems (at least one relevant item suggested).  
- Search results where user only needs one good answer.  


# Mean Rank (MR) and Median Rank

## 1. What it is
**Mean Rank (MR)** and **Median Rank** evaluate the average (or median) position of relevant items in the ranked list.  
Lower values are better.

## 2. Formula
For \( R_i \) = rank of relevant item in query \( i \):
$$
MR = \frac{1}{n} \sum_{i=1}^n R_i
$$

## 3. Interpretation
- MR = 1 → Relevant items always appear first.  
- Higher MR → Poorer ranking quality.  

## 4. Advantages
- Simple and interpretable.  
- Complements MAP and MRR.  

## 5. Limitations
- Sensitive to outliers (Mean Rank).  
- Ignores graded relevance.  

## 6. Use Cases
- Evaluating recommender systems.  
- Search engine ranking diagnostics.  


# Coverage

## 1. What it is
**Coverage** measures the proportion of items in the catalog that a recommender system is able to suggest.  
It indicates how well the model uses the full item space.

## 2. Formula
$$
\text{Coverage} = \frac{|\text{Items recommended at least once}|}{|\text{Total items}|}
$$

## 3. Interpretation
- Higher coverage = More diverse set of items recommended.  
- Low coverage = Model is biased toward a small subset.  

## 4. Advantages
- Important for recommender systems.  
- Prevents over-concentration on popular items.  

## 5. Limitations
- Does not measure relevance or ranking quality.  
- Can be artificially increased by recommending irrelevant items.  

## 6. Use Cases
- Movie, music, e-commerce recommendation systems.  
- Business applications where catalog exploration matters.  


# Diversity

## 1. What it is
**Diversity** measures how different the recommended items are from each other.  
It ensures that recommendations are not overly similar.

## 2. Formula
Pairwise diversity (using similarity \( sim \)):
$$
\text{Diversity} = 1 - \frac{2}{k(k-1)} \sum_{i < j} sim(item_i, item_j)
$$

## 3. Interpretation
- High diversity = Recommendations are varied.  
- Low diversity = Items are very similar (redundant).  

## 4. Advantages
- Improves user satisfaction (avoids redundancy).  
- Encourages discovery of new items.  

## 5. Limitations
- Trade-off: increasing diversity may reduce precision.  
- Requires a similarity measure (domain-dependent).  

## 6. Use Cases
- Music/movie recommendations.  
- E-commerce (ensuring product variety).  


# Novelty and Serendipity

## 1. What it is
- **Novelty**: Measures how "new" or unknown the recommended items are to the user (less popular items).  
- **Serendipity**: Measures how surprising but still useful the recommendations are.  

## 2. Interpretation
- High novelty → User sees fresh/unpopular items.  
- High serendipity → User receives unexpected but relevant suggestions.  

## 3. Advantages
- Encourages exploration.  
- Important for long-term user engagement.  

## 4. Limitations
- Hard to quantify objectively.  
- May recommend too many obscure items.  

## 5. Use Cases
- Personalized recommendation systems.  
- Online platforms balancing popular vs. niche content.  


# Expected Reciprocal Rank (ERR)

## 1. What it is
**Expected Reciprocal Rank (ERR)** improves on MRR by modeling the probability that a user stops at a given result.  
It accounts for graded relevance and user browsing behavior.

## 2. Formula
$$
ERR = \sum_{i=1}^n \frac{1}{i} \cdot P(\text{user stops at rank } i)
$$

Where stop probability depends on relevance levels of higher-ranked items.  

## 3. Interpretation
- ERR = 1 → Perfect ranking (user always satisfied at top result).  
- Lower ERR → Poorer ranking.  

## 4. Advantages
- Models realistic user satisfaction.  
- Handles graded relevance.  

## 5. Limitations
- More complex than MAP/MRR.  
- Requires relevance scores.  

## 6. Use Cases
- Search engine evaluation.  
- Large-scale IR benchmarks (e.g., TREC).  


# Precision–Recall Break-Even Point (PRBEP)

## 1. What it is
The **Precision–Recall Break-Even Point (PRBEP)** is the point where Precision equals Recall.  
It provides a single-number summary of PR trade-offs.

## 2. Formula
Find threshold where:
$$
\text{Precision} = \text{Recall}
$$

## 3. Interpretation
- Higher PRBEP → Better trade-off between Precision and Recall.  
- Useful when equal weight is given to both.  

## 4. Advantages
- Simple to interpret.  
- Classic IR metric.  

## 5. Limitations
- Only one point on PR curve (not overall performance).  
- Less common in modern ML, but still used in IR research.  

## 6. Use Cases
- Information retrieval benchmarks.  
- Binary classification with balanced Precision/Recall priorities.  


# Clustering Metrics

Clustering metrics evaluate how well a model groups similar data points together and separates dissimilar ones.  
Since clustering is often **unsupervised**, metrics are divided into:
- **Internal metrics**: use only the data and cluster assignments (e.g., Silhouette Score, Davies–Bouldin Index).  
- **External metrics**: require ground-truth labels to compare against (e.g., Adjusted Rand Index, Normalized Mutual Information).


# Silhouette Score

## 1. What it is
The **Silhouette Score** measures how similar each point is to its own cluster compared to other clusters.  
It combines both **cohesion** (tightness within a cluster) and **separation** (distance between clusters).

## 2. Formula
For each point \( i \):  
- \( a(i) \) = average distance to points in the same cluster.  
- \( b(i) \) = lowest average distance to points in another cluster.  

Silhouette for point \( i \):
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Overall Silhouette Score:
$$
S = \frac{1}{n} \sum_{i=1}^n s(i)
$$

## 3. Interpretation
- \( S = 1 \) → Perfect clustering (well-separated).  
- \( S = 0 \) → Overlapping clusters.  
- \( S < 0 \) → Misclassified points.  

## 4. Advantages
- Captures both intra-cluster cohesion and inter-cluster separation.  
- Works with any distance metric.  

## 5. Limitations
- Computationally expensive for large datasets.  
- Sensitive to distance metric choice.  
- May not work well for non-convex clusters.  

## 6. Use Cases
- Evaluating k-means, hierarchical, and density-based clustering.  
- Selecting the number of clusters (choose k with highest Silhouette Score).  


# Davies–Bouldin Index (DBI)

## 1. What it is
The **Davies–Bouldin Index (DBI)** measures the average similarity between each cluster and its most similar cluster.  
It evaluates clustering quality based on **intra-cluster similarity** and **inter-cluster separation**.

## 2. Formula
For cluster \( i \):  
- \( S_i \) = average distance of points in cluster \( i \) to its centroid.  
- \( M_{ij} \) = distance between centroids of clusters \( i \) and \( j \).  

Cluster similarity:
$$
R_{ij} = \frac{S_i + S_j}{M_{ij}}
$$

DBI:
$$
DBI = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} R_{ij}
$$

## 3. Interpretation
- Lower DBI = better clustering.  
- DBI = 0 → Ideal clustering (perfect separation).  

## 4. Advantages
- Considers both cohesion and separation.  
- Automated and easy to compute.  

## 5. Limitations
- Sensitive to noise and outliers.  
- Works best with convex clusters.  
- Results depend on distance metric choice.  

## 6. Use Cases
- Evaluating k-means and centroid-based clustering.  
- Model selection for optimal number of clusters.  


# Calinski–Harabasz Index (Variance Ratio Criterion)

## 1. What it is
The **Calinski–Harabasz Index (CHI)**, also known as the **Variance Ratio Criterion**, measures clustering quality by comparing between-cluster dispersion with within-cluster dispersion.

## 2. Formula
For \( k \) clusters, \( n \) samples:
$$
CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \cdot \frac{n - k}{k - 1}
$$

Where:  
- \( B_k \) = between-cluster dispersion matrix  
- \( W_k \) = within-cluster dispersion matrix  

## 3. Interpretation
- Higher CH → Better-defined clusters.  
- CH = 0 → No separation.  

## 4. Advantages
- Fast and efficient.  
- Works well for convex, spherical clusters.  

## 5. Limitations
- Biased towards algorithms with more clusters.  
- Sensitive to cluster shape and density.  

## 6. Use Cases
- Selecting the number of clusters in k-means.  
- Benchmarking different clustering algorithms.  


# Dunn Index

## 1. What it is
The **Dunn Index** evaluates clustering by maximizing the ratio between the minimum inter-cluster distance and the maximum intra-cluster diameter.

## 2. Formula
$$
D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_{k} \text{diam}(C_k)}
$$

Where:  
- \( d(C_i, C_j) \) = distance between clusters \( i \) and \( j \)  
- \( \text{diam}(C_k) \) = maximum distance between points in cluster \( k \)  

## 3. Interpretation
- Higher Dunn Index → Better clustering (well-separated, compact clusters).  

## 4. Advantages
- Captures both separation and compactness.  
- Good for detecting compact, well-separated clusters.  

## 5. Limitations
- Computationally expensive.  
- Very sensitive to noise and outliers.  

## 6. Use Cases
- Comparing clustering algorithms.  
- Quality assessment in small-to-medium datasets.  


# Adjusted Rand Index (ARI)

## 1. What it is
The **Adjusted Rand Index (ARI)** measures similarity between predicted clusters and true labels, adjusted for chance.  
It is an **external metric** (requires ground truth).

## 2. Formula
ARI is based on pair counting:
$$
ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}
$$

Where:  
- RI = Rand Index (proportion of correct pairwise agreements)  
- \( E[RI] \) = expected RI under random labeling  

## 3. Interpretation
- ARI = 1 → Perfect agreement.  
- ARI = 0 → Random clustering.  
- ARI < 0 → Worse than random.  

## 4. Advantages
- Adjusts for chance, more reliable than RI.  
- Widely used in clustering benchmarks.  

## 5. Limitations
- Requires ground-truth labels.  
- Sensitive to number of clusters.  

## 6. Use Cases
- Benchmarking clustering against known categories.  
- NLP, bioinformatics, and image segmentation tasks.  


# Normalized Mutual Information (NMI)

## 1. What it is
The **Normalized Mutual Information (NMI)** measures the mutual dependence between cluster assignments and ground-truth labels.  
It is normalized to range [0,1].

## 2. Formula
$$
NMI = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}
$$

Where:  
- \( I(Y; C) \) = mutual information between true labels \( Y \) and clusters \( C \)  
- \( H(\cdot) \) = entropy  

## 3. Interpretation
- NMI = 1 → Perfect match.  
- NMI = 0 → No correlation.  

## 4. Advantages
- Symmetric and normalized.  
- Works well for comparing clusterings with different numbers of clusters.  

## 5. Limitations
- Requires ground-truth labels.  
- Sensitive to noisy labels.  

## 6. Use Cases
- Clustering evaluation in NLP (topic modeling).  
- Image segmentation validation.  


# Homogeneity, Completeness, and V-Measure

## 1. What it is
These three related metrics evaluate clustering quality compared to ground truth:
- **Homogeneity**: Each cluster contains only members of a single class.  
- **Completeness**: All members of a class are assigned to the same cluster.  
- **V-Measure**: Harmonic mean of Homogeneity and Completeness.  

## 2. Formula
- Homogeneity:
$$
h = 1 - \frac{H(C|Y)}{H(C)}
$$

- Completeness:
$$
c = 1 - \frac{H(Y|C)}{H(Y)}
$$

- V-Measure:
$$
V = 2 \cdot \frac{h \cdot c}{h + c}
$$

## 3. Interpretation
- h = 1 → Perfect homogeneity.  
- c = 1 → Perfect completeness.  
- V = 1 → Perfect overall clustering.  

## 4. Advantages
- Easy to interpret.  
- Balances cluster purity and class completeness.  

## 5. Limitations
- Requires ground-truth labels.  
- Sensitive to class imbalance.  

## 6. Use Cases
- NLP topic modeling.  
- Multi-class clustering validation.  


# Purity

## 1. What it is
**Purity** measures the extent to which clusters contain a single class.  
It is a simple external evaluation metric.

## 2. Formula
$$
\text{Purity} = \frac{1}{n} \sum_{k} \max_j |C_k \cap Y_j|
$$

Where:  
- \( C_k \) = cluster \( k \)  
- \( Y_j \) = ground-truth class \( j \)  

## 3. Interpretation
- Purity = 1 → Perfect clustering.  
- Higher values = more homogeneous clusters.  

## 4. Advantages
- Intuitive and simple.  
- Easy to compute.  

## 5. Limitations
- Biased towards more clusters (trivial solution = 1 cluster per point).  
- Does not consider completeness.  

## 6. Use Cases
- Initial clustering evaluation.  
- Educational examples and simple benchmarks.  


# Modularity (for Community Detection)

## 1. What it is
**Modularity** evaluates clustering (or community detection) in graphs by comparing the density of edges inside clusters to the density expected in a random graph.

## 2. Formula
$$
Q = \frac{1}{2m} \sum_{i,j} \Big[ A_{ij} - \frac{k_i k_j}{2m} \Big] \delta(c_i, c_j)
$$

Where:  
- \( A_{ij} \) = adjacency matrix (1 if edge exists, 0 otherwise)  
- \( k_i \) = degree of node \( i \)  
- \( m \) = total number of edges  
- \( \delta(c_i, c_j) = 1 \) if nodes \( i, j \) in same cluster, else 0  

## 3. Interpretation
- Modularity close to 1 → Strong community structure.  
- Modularity ≈ 0 → No better than random.  

## 4. Advantages
- Standard for graph clustering.  
- Intuitive: compares structure to random networks.  

## 5. Limitations
- Resolution limit (may miss small communities).  
- Not suitable outside graph-based clustering.  

## 6. Use Cases
- Social network analysis.  
- Biological networks (protein interactions).  
- Community detection in large graphs.  


# Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC)

## 1. What it is
**BIC** and **AIC** are statistical criteria used to evaluate model-based clustering (e.g., Gaussian Mixture Models).  
They balance model fit (likelihood) with model complexity (number of parameters).

## 2. Formulas
- AIC:
$$
AIC = 2k - 2\ln(L)
$$

- BIC:
$$
BIC = k \ln(n) - 2\ln(L)
$$

Where:  
- \( k \) = number of parameters  
- \( n \) = number of samples  
- \( L \) = likelihood  

## 3. Interpretation
- Lower AIC/BIC → Better model.  
- BIC penalizes complexity more strongly than AIC.  

## 4. Advantages
- Helps select optimal number of clusters.  
- Rigorous statistical foundation.  

## 5. Limitations
- Only applies to probabilistic clustering (e.g., GMM).  
- Assumes likelihood estimation is correct.  

## 6. Use Cases
- Gaussian Mixture Models.  
- Model selection in statistical clustering.  


# Entropy (Cluster Label Distribution)

## 1. What it is
**Entropy** measures class distribution within clusters.  
It evaluates how “pure” clusters are with respect to ground-truth labels.

## 2. Formula
For cluster \( C_k \):
$$
H(C_k) = - \sum_j p_{k,j} \log(p_{k,j})
$$

Overall entropy:
$$
H = \sum_k \frac{|C_k|}{n} H(C_k)
$$

Where:  
- \( p_{k,j} \) = proportion of class \( j \) in cluster \( k \).  

## 3. Interpretation
- Lower entropy → More homogeneous clusters.  
- Entropy = 0 → Perfect purity.  

## 4. Advantages
- Intuitive and interpretable.  
- Complements Purity.  

## 5. Limitations
- Biased toward more clusters.  
- Requires ground-truth labels.  

## 6. Use Cases
- Comparing clustering results to ground truth.  
- Text and topic clustering validation.  


# Gap Statistic

## 1. What it is
The **Gap Statistic** helps determine the optimal number of clusters by comparing within-cluster dispersion to that of a reference (random) dataset.

## 2. Formula
For cluster number \( k \):
$$
Gap(k) = E^*[\log(W_k)] - \log(W_k)
$$

Where:  
- \( W_k \) = within-cluster dispersion for \( k \) clusters  
- \( E^* \) = expectation under reference distribution  

## 3. Interpretation
- Larger Gap(k) → Better clustering compared to random.  
- Choose \( k \) that maximizes the Gap.  

## 4. Advantages
- Provides principled way to choose cluster number.  
- Widely used in practice.  

## 5. Limitations
- Computationally expensive (requires simulation).  
- Sensitive to reference distribution choice.  

## 6. Use Cases
- Choosing number of clusters in k-means.  
- General model selection for clustering.  


# Time-Series Forecasting Metrics

Time-series forecasting metrics evaluate the accuracy of predicted values over time.  
They are designed to handle sequential dependencies and are often sensitive to scale, seasonality, and trend.  
Some metrics are **scale-dependent** (like MAE, RMSE), while others are **scale-independent** (like MAPE, MASE), which allows fair comparison across datasets.


# Mean Absolute Error (MAE)

## 1. What it is
The **Mean Absolute Error (MAE)** measures the average magnitude of errors in predictions, without considering their direction.  
It is one of the most interpretable and widely used metrics for time-series.

## 2. Formula
$$
MAE = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|
$$

Where:  
- \( y_t \) = actual value at time \( t \)  
- \( \hat{y}_t \) = predicted value  

## 3. Interpretation
- MAE = 0 → Perfect forecast.  
- Lower MAE → Better model.  
- Expressed in the same units as the data.  

## 4. Advantages
- Simple and intuitive.  
- Robust to outliers compared to squared error metrics.  

## 5. Limitations
- Scale-dependent (cannot compare across datasets).  
- Does not penalize large errors as strongly as RMSE.  

## 6. Use Cases
- Forecasting electricity demand, stock prices, sales, or weather.  
- Baseline metric for model comparison.  


# Mean Squared Error (MSE)

## 1. What it is
The **Mean Squared Error (MSE)** measures the average squared difference between actual and predicted values.  
It penalizes large errors more strongly than MAE.

## 2. Formula
$$
MSE = \frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2
$$

## 3. Interpretation
- MSE = 0 → Perfect forecast.  
- Larger errors have disproportionately higher impact.  

## 4. Advantages
- Differentiable (useful in optimization).  
- Penalizes large errors, encouraging cautious predictions.  

## 5. Limitations
- Scale-dependent.  
- Sensitive to outliers.  

## 6. Use Cases
- Model training and optimization.  
- Evaluating forecasts where large errors are costly.  


# Root Mean Squared Error (RMSE)

## 1. What it is
The **Root Mean Squared Error (RMSE)** is the square root of MSE, bringing error back to the same unit as the original data.  
It is widely used in time-series forecasting.

## 2. Formula
$$
RMSE = \sqrt{ \frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2 }
$$

## 3. Interpretation
- RMSE = 0 → Perfect forecast.  
- More sensitive to large errors than MAE.  

## 4. Advantages
- Same units as data, easier to interpret than MSE.  
- Standard in many forecasting competitions.  

## 5. Limitations
- Still scale-dependent.  
- Sensitive to outliers.  

## 6. Use Cases
- Weather prediction.  
- Energy consumption forecasting.  
- Kaggle forecasting competitions.  


# Mean Absolute Percentage Error (MAPE)

## 1. What it is
The **Mean Absolute Percentage Error (MAPE)** expresses forecast accuracy as a percentage, making it scale-independent.

## 2. Formula
$$
MAPE = \frac{100}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$

## 3. Interpretation
- MAPE = 0% → Perfect forecast.  
- Example: MAPE = 10% → On average, forecast is off by 10%.  

## 4. Advantages
- Intuitive and comparable across datasets.  
- Expressed as a percentage.  

## 5. Limitations
- Undefined when \( y_t = 0 \).  
- Biased when values are very small.  

## 6. Use Cases
- Business and finance forecasting (sales, revenue).  
- Benchmarking models across industries.  


# Symmetric Mean Absolute Percentage Error (sMAPE)

## 1. What it is
The **sMAPE** modifies MAPE to be symmetric and bounded between 0% and 200%.  
It avoids issues when \( y_t \) is close to 0.

## 2. Formula
$$
sMAPE = \frac{100}{n} \sum_{t=1}^n \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|)/2}
$$

## 3. Interpretation
- sMAPE = 0% → Perfect forecast.  
- Bounded: 0% ≤ sMAPE ≤ 200%.  

## 4. Advantages
- More stable than MAPE.  
- Scale-independent.  

## 5. Limitations
- Still biased if predictions are systematically low/high.  
- Less intuitive than MAPE.  

## 6. Use Cases
- Forecasting competitions (e.g., M4).  
- Business and demand forecasting.  


# Mean Absolute Scaled Error (MASE)

## 1. What it is
The **MASE** compares forecast accuracy to a naïve benchmark (like last-value forecast).  
It is scale-independent and robust.

## 2. Formula
$$
MASE = \frac{\frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|}{\frac{1}{n-1} \sum_{t=2}^n |y_t - y_{t-1}|}
$$

## 3. Interpretation
- MASE = 1 → Forecast performs same as naïve benchmark.  
- MASE < 1 → Better than naïve forecast.  
- MASE > 1 → Worse than naïve forecast.  

## 4. Advantages
- Scale-free.  
- Robust to seasonality if seasonal benchmark is used.  

## 5. Limitations
- Requires a sensible benchmark.  
- Less intuitive than MAPE.  

## 6. Use Cases
- Comparing forecasting methods across datasets.  
- Time-series competitions (M4, M5).  


# Theil’s U Statistic

## 1. What it is
**Theil’s U** compares forecast accuracy to a naïve model (e.g., random walk or last observation).  
It is a relative measure.

## 2. Formula
$$
U = \frac{\sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}}{\sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - y_{t-1})^2}}
$$

## 3. Interpretation
- U < 1 → Forecast better than naïve.  
- U = 1 → Same as naïve.  
- U > 1 → Worse than naïve.  

## 4. Advantages
- Easy to interpret relative performance.  
- Standard in econometrics and forecasting research.  

## 5. Limitations
- Benchmark choice matters.  
- Less intuitive than MAE or RMSE.  

## 6. Use Cases
- Macroeconomic forecasting.  
- Financial time-series analysis.  


# Mean Forecast Error (MFE, Bias)

## 1. What it is
The **Mean Forecast Error (MFE)**, also known as **Bias**, measures the average signed error.  
It shows whether forecasts are systematically over- or under-estimated.

## 2. Formula
$$
MFE = \frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)
$$

## 3. Interpretation
- MFE = 0 → Unbiased forecast.  
- Positive MFE → Underestimation.  
- Negative MFE → Overestimation.  

## 4. Advantages
- Simple diagnostic of bias.  
- Complements error magnitude metrics.  

## 5. Limitations
- Positive and negative errors can cancel out.  
- Not a good standalone accuracy metric.  

## 6. Use Cases
- Checking bias in demand forecasting.  
- Evaluating systematic trends in model errors.  


# Mean Squared Logarithmic Error (MSLE)

## 1. What it is
The **MSLE** is useful when the scale of data varies greatly.  
It penalizes underestimation more than overestimation and works on log-transformed values.

## 2. Formula
$$
MSLE = \frac{1}{n} \sum_{t=1}^n \left( \log(1 + y_t) - \log(1 + \hat{y}_t) \right)^2
$$

## 3. Interpretation
- MSLE = 0 → Perfect forecast.  
- Emphasizes relative error instead of absolute error.  

## 4. Advantages
- Good for data with exponential growth.  
- Less sensitive to large outliers.  

## 5. Limitations
- Requires positive values.  
- Harder to interpret than MAE or RMSE.  

## 6. Use Cases
- Population growth forecasts.  
- Economic time-series with compounding growth.  


# Weighted Absolute Percentage Error (WAPE)

## 1. What it is
**WAPE** is a normalized error metric that scales MAE by the sum of actual values.  
It is sometimes referred to as "overall percentage error".

## 2. Formula
$$
WAPE = \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{\sum_{t=1}^n |y_t|}
$$

## 3. Interpretation
- WAPE = 0 → Perfect forecast.  
- Often expressed as a percentage.  

## 4. Advantages
- Simple and scale-independent.  
- Similar interpretation to MAPE.  

## 5. Limitations
- Can overweight small denominators (like MAPE).  
- Sensitive to extreme values.  

## 6. Use Cases
- Retail demand forecasting.  
- Large-scale sales forecasting.  


# Median Absolute Error (MedAE)

## 1. What it is
The **Median Absolute Error (MedAE)** is the median of absolute errors between predictions and actual values.  
It is more robust to outliers than MAE.

## 2. Formula
$$
MedAE = \text{median}(|y_t - \hat{y}_t|)
$$

## 3. Advantages
- Robust to outliers.  
- Simple to interpret.  

## 4. Limitations
- Less smooth than MAE (not differentiable).  

## 5. Use Cases
- Financial or demand forecasting where outliers exist.  


# Root Mean Squared Percentage Error (RMSPE)

## 1. What it is
The **RMSPE** is the square root of the Mean Squared Percentage Error, penalizing large percentage errors more heavily than MAPE.

## 2. Formula
$$
RMSPE = \sqrt{\frac{100^2}{n} \sum_{t=1}^n \left(\frac{y_t - \hat{y}_t}{y_t}\right)^2}
$$

## 3. Advantages
- Scale-independent.  
- More sensitive to large percentage errors.  

## 4. Limitations
- Undefined when \( y_t = 0 \).  
- More affected by small denominators than MAPE.  

## 5. Use Cases
- Forecasting with high variance in demand or sales.  


# Root Mean Squared Percentage Error (RMSPE)

## 1. What it is
The **RMSPE** is the square root of the Mean Squared Percentage Error, penalizing large percentage errors more heavily than MAPE.

## 2. Formula
$$
RMSPE = \sqrt{\frac{100^2}{n} \sum_{t=1}^n \left(\frac{y_t - \hat{y}_t}{y_t}\right)^2}
$$

## 3. Advantages
- Scale-independent.  
- More sensitive to large percentage errors.  

## 4. Limitations
- Undefined when \( y_t = 0 \).  
- More affected by small denominators than MAPE.  

## 5. Use Cases
- Forecasting with high variance in demand or sales.  


# Root Mean Squared Scaled Error (RMSSE)

## 1. What it is
The **RMSSE** is a scaled version of RMSE relative to a naïve benchmark (like last-observation).  
It was popularized in the **M5 forecasting competition**.

## 2. Formula
$$
RMSSE = \sqrt{ \frac{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}{\frac{1}{n-1} \sum_{t=2}^n (y_t - y_{t-1})^2} }
$$

## 3. Advantages
- Scale-independent.  
- Allows fair comparison across series.  

## 4. Limitations
- Requires benchmark choice.  
- More complex than RMSE.  

## 5. Use Cases
- Model comparison across multiple time-series.  
- Competitions and large-scale benchmarks.  


# Mean Arctangent Absolute Percentage Error (MAAPE)

## 1. What it is
The **MAAPE** improves over MAPE by using an arctangent transformation, avoiding extreme percentage errors when actuals are near zero.

## 2. Formula
$$
MAAPE = \frac{1}{n} \sum_{t=1}^n \arctan \left( \left| \frac{y_t - \hat{y}_t}{y_t} \right| \right)
$$

## 3. Advantages
- Handles division by zero better than MAPE.  
- Bounded and more stable.  

## 4. Limitations
- Less intuitive than MAPE (measured in radians).  

## 5. Use Cases
- Forecasting datasets with zeros or small denominators.  
- Financial and demand series.  


# Pinball Loss (Quantile Loss)

## 1. What it is
**Pinball Loss** is used in quantile forecasting, measuring accuracy for predicting conditional quantiles.  
It’s the standard loss for probabilistic forecasts.

## 2. Formula
For quantile \( \tau \):
$$
L_\tau(y_t, \hat{y}_t) = 
\begin{cases} 
\tau (y_t - \hat{y}_t), & y_t \geq \hat{y}_t \\
(1 - \tau)(\hat{y}_t - y_t), & y_t < \hat{y}_t
\end{cases}
$$

## 3. Advantages
- Supports probabilistic and interval forecasts.  
- Standard in quantile regression.  

## 4. Limitations
- Less intuitive than MAE/ RMSE.  

## 5. Use Cases
- Probabilistic demand forecasting.  
- Quantile regression models.  


# Prediction Interval Coverage Probability (PICP)

## 1. What it is
The **PICP** measures how often the actual values fall inside the predicted interval (e.g., 95% interval).

## 2. Formula
$$
PICP = \frac{1}{n} \sum_{t=1}^n \mathbf{1}\{ y_t \in [L_t, U_t] \}
$$

Where:  
- \( L_t, U_t \) = lower and upper bounds of prediction interval.  

## 3. Advantages
- Evaluates reliability of uncertainty estimates.  

## 4. Limitations
- Does not penalize interval width.  

## 5. Use Cases
- Forecasting with uncertainty estimation.  
- Energy and financial forecasting.  


# Interval Score (IS)

## 1. What it is
The **Interval Score** balances interval width and coverage, rewarding narrow but reliable prediction intervals.

## 2. Formula
For interval \([L_t, U_t]\) at level \( 1 - \alpha \):
$$
IS = (U_t - L_t) + \frac{2}{\alpha} (L_t - y_t) \mathbf{1}\{y_t < L_t\} + \frac{2}{\alpha} (y_t - U_t) \mathbf{1}\{y_t > U_t\}
$$

## 3. Advantages
- Balances sharpness and calibration.  
- Encourages informative intervals.  

## 4. Limitations
- Requires interval forecasts.  

## 5. Use Cases
- Probabilistic forecasting.  
- Energy, finance, epidemiology.  


# Computer Vision Metrics

Computer vision metrics evaluate the performance of models in tasks such as **image classification, object detection, image segmentation, and image generation**.  
These metrics often focus on **spatial overlap**, **image quality**, and **perceptual similarity**, making them different from standard regression/classification metrics.


# Intersection over Union (IoU)

## 1. What it is
**Intersection over Union (IoU)**, also known as the Jaccard Index, measures the overlap between predicted and ground-truth regions (bounding boxes or segmentation masks).

## 2. Formula
$$
IoU = \frac{|A \cap B|}{|A \cup B|}
$$

Where:  
- \( A \) = predicted region  
- \( B \) = ground-truth region  

## 3. Interpretation
- IoU = 1 → Perfect overlap.  
- IoU = 0 → No overlap.  
- Common thresholds: IoU ≥ 0.5 or 0.75 for “correct” detection.  

## 4. Advantages
- Intuitive and interpretable.  
- Standard metric for detection tasks.  

## 5. Limitations
- Sensitive to small localization errors.  
- Does not capture boundary quality in detail.  

## 6. Use Cases
- Object detection benchmarks (Pascal VOC, COCO).  
- Semantic and instance segmentation.  
- Evaluating bounding box quality.  


# Dice Coefficient (F1 for Segmentation)

## 1. What it is
The **Dice Coefficient** measures the overlap between predicted and ground-truth regions, similar to IoU but weighted differently.  
It is equivalent to the F1-Score in segmentation.

## 2. Formula
$$
Dice = \frac{2|A \cap B|}{|A| + |B|}
$$

## 3. Interpretation
- Dice = 1 → Perfect overlap.  
- Dice = 0 → No overlap.  

## 4. Advantages
- More sensitive than IoU for small objects.  
- Widely used in medical image segmentation.  

## 5. Limitations
- Like IoU, does not capture fine boundary details.  

## 6. Use Cases
- Medical imaging (tumor segmentation).  
- Semantic and instance segmentation.  


# Mean Average Precision (mAP)

## 1. What it is
The **Mean Average Precision (mAP)** is the mean of Average Precision (AP) across all classes, commonly used for object detection.

## 2. Formula
For a class \( c \):
$$
AP_c = \int_0^1 P_c(R) \, dR
$$

Then:
$$
mAP = \frac{1}{C} \sum_{c=1}^C AP_c
$$

Where:  
- \( P_c(R) \) = precision-recall curve for class \( c \)  
- \( C \) = number of classes  

## 3. Interpretation
- mAP = 1 → Perfect detection across all classes.  
- Benchmarks often report mAP@0.5 or mAP@[0.5:0.95].  

## 4. Advantages
- Combines precision and recall.  
- Standard metric for detection challenges.  

## 5. Limitations
- Computationally expensive.  
- Threshold choice (IoU cutoff) affects results.  

## 6. Use Cases
- Object detection benchmarks (Pascal VOC, COCO, YOLO).  
- Multi-class detection tasks.  


# Peak Signal-to-Noise Ratio (PSNR)

## 1. What it is
The **PSNR** measures the quality of reconstructed images (e.g., after compression, super-resolution) compared to original images.

## 2. Formula
$$
PSNR = 10 \log_{10} \left( \frac{MAX^2}{MSE} \right)
$$

Where:  
- \( MAX \) = maximum possible pixel value (e.g., 255 for 8-bit images)  
- \( MSE \) = mean squared error between images  

## 3. Interpretation
- Higher PSNR → Better quality.  
- Typical values: 20–40 dB.  

## 4. Advantages
- Simple, fast to compute.  

## 5. Limitations
- Poor correlation with human perception.  

## 6. Use Cases
- Image compression evaluation.  
- Image super-resolution.  
- Denoising tasks.  


# Structural Similarity Index (SSIM)

## 1. What it is
The **SSIM** measures perceptual similarity between two images, considering luminance, contrast, and structure.

## 2. Formula
$$
SSIM(x,y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where:  
- \( \mu \) = mean intensity  
- \( \sigma \) = variance  
- \( \sigma_{xy} \) = covariance  

## 3. Interpretation
- SSIM = 1 → Identical images.  
- SSIM = 0 → No similarity.  

## 4. Advantages
- Correlates well with human vision.  
- Better than PSNR for perceptual quality.  

## 5. Limitations
- Computationally more complex.  
- May not capture color/texture differences.  

## 6. Use Cases
- GAN evaluation.  
- Image quality assessment.  
- Video compression quality.  


# Fréchet Inception Distance (FID)

## 1. What it is
**FID** measures the distance between distributions of real and generated images, based on features from an Inception network.

## 2. Formula
$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

Where:  
- \( \mu_r, \Sigma_r \) = mean and covariance of real images  
- \( \mu_g, \Sigma_g \) = mean and covariance of generated images  

## 3. Interpretation
- Lower FID = Better quality and diversity.  

## 4. Advantages
- Captures both quality and diversity.  
- Standard metric for GANs.  

## 5. Limitations
- Sensitive to feature extractor choice.  
- Requires many samples for stability.  

## 6. Use Cases
- GAN evaluation.  
- Image generation benchmarks.  


# Inception Score (IS)

## 1. What it is
**Inception Score** evaluates generated image quality and diversity using a pre-trained classifier.

## 2. Formula
$$
IS = \exp \left( \mathbb{E}_x \, KL(p(y|x) \| p(y)) \right)
$$

Where:  
- \( p(y|x) \) = label distribution for generated image \( x \)  
- \( p(y) \) = marginal label distribution  

## 3. Interpretation
- Higher IS = Clear, diverse images.  

## 4. Advantages
- Simple and widely used in early GAN work.  

## 5. Limitations
- Ignores realism relative to real data.  
- Can be gamed by mode collapse with confident outputs.  

## 6. Use Cases
- GAN evaluation (early benchmarks).  


# Kernel Inception Distance (KID)

## 1. What it is
**KID** is an alternative to FID that uses polynomial kernels to measure similarity between real and generated images.  
Unlike FID, it provides an **unbiased estimator**.

## 2. Advantages
- More stable with small sample sizes.  
- Unbiased compared to FID.  

## 3. Limitations
- Less widely adopted than FID.  

## 4. Use Cases
- GAN evaluation.  
- Image generation tasks with limited data.  


# Pixel Accuracy (PA)

## 1. What it is
**Pixel Accuracy (PA)** measures the percentage of correctly classified pixels in an image segmentation task.

## 2. Formula
$$
PA = \frac{\sum_i n_{ii}}{\sum_i \sum_j n_{ij}}
$$

Where:  
- \( n_{ii} \) = number of correctly classified pixels for class \( i \)  
- \( n_{ij} \) = number of pixels of class \( i \) predicted as class \( j \)  

## 3. Advantages
- Simple and intuitive.  

## 4. Limitations
- Biased toward majority classes (background dominates).  

## 5. Use Cases
- Semantic segmentation (baseline metric).  


# Mean Pixel Accuracy (mPA)

## 1. What it is
**Mean Pixel Accuracy (mPA)** is the average pixel accuracy across classes, reducing bias from majority classes.

## 2. Formula
$$
mPA = \frac{1}{C} \sum_{i=1}^C \frac{n_{ii}}{\sum_j n_{ij}}
$$

Where \( C \) = number of classes.  

## 3. Advantages
- Balances evaluation across classes.  

## 4. Limitations
- Still ignores boundary quality.  

## 5. Use Cases
- Multi-class segmentation evaluation.  


# Mean Intersection over Union (mIoU)

## 1. What it is
**mIoU** averages the Intersection over Union across all classes, making it a standard metric for segmentation challenges.

## 2. Formula
$$
mIoU = \frac{1}{C} \sum_{i=1}^C \frac{n_{ii}}{\sum_j n_{ij} + \sum_j n_{ji} - n_{ii}}
$$

## 3. Advantages
- Standard, widely reported.  
- Balances across classes.  

## 4. Limitations
- Sensitive to class imbalance.  

## 5. Use Cases
- Semantic segmentation benchmarks (Pascal VOC, Cityscapes).  


# Panoptic Quality (PQ)

## 1. What it is
**Panoptic Quality (PQ)** is used in **panoptic segmentation**, combining detection quality and segmentation quality.

## 2. Formula
$$
PQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP| + \tfrac{1}{2}|FP| + \tfrac{1}{2}|FN|}
$$

Where:  
- \( p \) = predicted segments  
- \( g \) = ground-truth segments  
- TP, FP, FN = true/false positives/negatives  

## 3. Advantages
- Unified metric for detection + segmentation.  

## 4. Limitations
- More complex than IoU/mIoU.  

## 5. Use Cases
- Panoptic segmentation tasks (Cityscapes, COCO Panoptic).  


# Boundary IoU / Boundary F1 Score

## 1. What it is
These metrics evaluate how well the predicted boundaries align with ground-truth boundaries, especially for fine segmentation.

## 2. Formula
- Boundary IoU:  
$$
Boundary\ IoU = \frac{|Boundary(P) \cap Boundary(G)|}{|Boundary(P) \cup Boundary(G)|}
$$  

- Boundary F1:  
Computes precision and recall of predicted vs. true boundary pixels, then harmonic mean.  

## 3. Advantages
- Focuses on boundary quality (important in medical imaging).  

## 4. Limitations
- Sensitive to annotation noise.  

## 5. Use Cases
- Medical segmentation (tumor borders).  
- Fine object segmentation tasks.  


# Learned Perceptual Image Patch Similarity (LPIPS)

## 1. What it is
**LPIPS** measures perceptual similarity using deep features from neural networks.  
Instead of pixel-level differences, it compares deep embeddings.

## 2. Formula
Given network \( F \) with layers \( l \):
$$
LPIPS(x,y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} ||w_l \odot (F_l(x)_{hw} - F_l(y)_{hw})||_2^2
$$

## 3. Advantages
- Strong correlation with human perception.  
- Widely adopted in GAN evaluation.  

## 4. Limitations
- Requires pre-trained deep networks.  
- Computationally more expensive.  

## 5. Use Cases
- Image generation quality (GANs, diffusion models).  
- Super-resolution and image editing evaluation.  


# Natural Language Processing (NLP) Metrics

NLP metrics evaluate model performance in tasks such as **machine translation, text summarization, language modeling, and text generation**.  
They can be divided into:  
- **Lexical overlap metrics** (e.g., BLEU, ROUGE, METEOR)  
- **Probability-based metrics** (e.g., Perplexity)  
- **Embedding-based metrics** (e.g., BERTScore, MoverScore)  


# BLEU (Bilingual Evaluation Understudy)

## 1. What it is
**BLEU** measures the similarity between a machine-generated text and one or more reference texts using n-gram overlap.  
It is one of the first widely adopted MT evaluation metrics.

## 2. Formula
For n-gram precision:
$$
p_n = \frac{\sum_{C \in Candidates} \sum_{n\text{-gram} \in C} Count_{clip}(n\text{-gram})}{\sum_{C \in Candidates} \sum_{n\text{-gram} \in C} Count(n\text{-gram})}
$$

BLEU score:
$$
BLEU = BP \cdot \exp \left( \sum_{n=1}^N w_n \log p_n \right)
$$

Where:  
- \( BP \) = brevity penalty (to penalize short outputs)  
- \( w_n \) = weight for n-grams (usually uniform)  

## 3. Interpretation
- BLEU = 1 → Perfect match with reference.  
- BLEU = 0 → No n-gram overlap.  

## 4. Advantages
- Simple, fast, language-independent.  
- Standard for MT benchmarking.  

## 5. Limitations
- Ignores synonyms and meaning.  
- Penalizes creative paraphrasing.  
- Weak correlation with human judgment for long texts.  

## 6. Use Cases
- Machine translation.  
- Summarization.  
- Text generation tasks.  


# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

## 1. What it is
**ROUGE** measures overlap between machine-generated text and reference text, focusing on recall.  
Variants include:  
- **ROUGE-N**: n-gram overlap (recall).  
- **ROUGE-L**: longest common subsequence.  
- **ROUGE-S**: skip-bigram overlap.  

## 2. Formula
For ROUGE-N:
$$
ROUGE\text{-}N = \frac{\sum_{S \in References} \sum_{n\text{-gram} \in S} Count_{match}(n\text{-gram})}{\sum_{S \in References} \sum_{n\text{-gram} \in S} Count(n\text{-gram})}
$$

## 3. Advantages
- Good for summarization.  
- Captures recall well.  

## 4. Limitations
- Ignores synonyms and meaning.  
- Sensitive to length.  

## 5. Use Cases
- Text summarization evaluation.  
- MT quality benchmarking.  


# METEOR

## 1. What it is
**METEOR** improves BLEU by considering stemming, synonyms, and word order.  

## 2. Formula
Combines:
- Unigram precision and recall.  
- Fragmentation penalty (penalizes disordered matches).  

## 3. Advantages
- Better correlation with human judgment than BLEU.  
- Handles synonyms and paraphrasing.  

## 4. Limitations
- More complex to compute.  
- Still surface-form dependent.  

## 5. Use Cases
- Machine translation.  
- Summarization evaluation.  


# Perplexity

## 1. What it is
**Perplexity** measures how well a probabilistic language model predicts a sequence.  
Lower perplexity = better predictive power.

## 2. Formula
$$
PP = 2^{-\frac{1}{n} \sum_{i=1}^n \log_2 P(w_i)}
$$

Where \( P(w_i) \) = predicted probability of word \( w_i \).  

## 3. Advantages
- Standard for language models.  
- Interpretable: lower = better.  

## 4. Limitations
- Not always aligned with human judgment.  
- Cannot be used when probabilities are not available.  

## 5. Use Cases
- Language modeling (n-grams, RNNs, Transformers).  
- Speech recognition.  


# BERTScore

## 1. What it is
**BERTScore** uses contextual embeddings (BERT) to measure similarity between candidate and reference sentences.  

## 2. Formula
For each token in candidate, find max cosine similarity with reference embeddings, then average.  
Provides precision, recall, and F1-like scores.  

## 3. Advantages
- Captures semantic similarity.  
- Works with synonyms and paraphrasing.  

## 4. Limitations
- Requires heavy pre-trained models.  
- Computationally expensive.  

## 5. Use Cases
- Summarization evaluation.  
- Machine translation.  
- Text generation tasks.  


# MoverScore

## 1. What it is
**MoverScore** extends BERTScore using Earth Mover’s Distance on word embeddings, capturing semantic similarity and word alignment.  

## 2. Advantages
- Better correlation with human judgment.  
- Handles paraphrasing and reordering.  

## 3. Limitations
- Expensive to compute.  
- Still embedding-model dependent.  

## 4. Use Cases
- Text summarization.  
- MT and paraphrase evaluation.  


# ChrF / ChrF++

## 1. What it is
**ChrF** uses character n-gram F-score (instead of words), making it more robust to morphology and spelling variations.  
**ChrF++** adds word n-grams.  

## 2. Advantages
- Strong for morphologically rich languages.  
- Language-independent.  

## 3. Limitations
- May reward superficial overlap without semantic meaning.  

## 4. Use Cases
- MT evaluation for agglutinative languages (e.g., Turkish, Finnish).  


# GLEU

## 1. What it is
**GLEU** (Google-BLEU) is a variant of BLEU that balances precision and recall better.  

## 2. Advantages
- Simpler than BLEU.  
- Better correlation with human judgment in some tasks.  

## 3. Limitations
- Less widely used than BLEU/ROUGE.  

## 4. Use Cases
- Machine translation.  


# Exact Match (EM)

## 1. What it is
**Exact Match** measures whether the prediction matches the reference exactly.  

## 2. Advantages
- Very strict, easy to compute.  

## 3. Limitations
- Too harsh: paraphrased correct answers get 0.  

## 4. Use Cases
- QA benchmarks (e.g., SQuAD).  


# F1 Score (in QA/NLP)

## 1. What it is
Measures token-level overlap between prediction and reference answers.  
Used in extractive QA tasks.  

## 2. Formula
$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

## 3. Use Cases
- Question answering (SQuAD, Natural Questions).  


# Translation Edit Rate (TER)

## 1. What it is
**TER** measures the number of edits required to transform a system output into a reference translation.  
Edits include insertions, deletions, substitutions, and shifts of word sequences.

## 2. Formula
$$
TER = \frac{\text{Number of edits}}{\text{Average number of reference words}}
$$

## 3. Advantages
- Intuitive: based on human editing effort.  
- Strong correlation with post-editing work.  

## 4. Limitations
- Sensitive to exact wording.  
- Ignores synonyms/paraphrasing.  

## 5. Use Cases
- Machine translation evaluation.  


# Word Error Rate (WER)

## 1. What it is
**WER** is widely used in speech recognition.  
It computes the Levenshtein distance (edit distance) between predicted and reference transcriptions.

## 2. Formula
$$
WER = \frac{S + D + I}{N}
$$

Where:  
- \( S \) = substitutions  
- \( D \) = deletions  
- \( I \) = insertions  
- \( N \) = total words in reference  

## 3. Advantages
- Standard in ASR benchmarks.  
- Easy to interpret.  

## 4. Limitations
- Very strict: small word differences heavily penalized.  
- Ignores semantic similarity.  

## 5. Use Cases
- Automatic Speech Recognition (ASR).  
- Spoken language evaluation.  


# Character Error Rate (CER)

## 1. What it is
**CER** is the character-level equivalent of WER, useful for languages with long words or when word segmentation is difficult.

## 2. Formula
Similar to WER, but with characters instead of words:
$$
CER = \frac{S + D + I}{N}
$$

## 3. Advantages
- More fine-grained than WER.  
- Works well for character-based languages (e.g., Chinese).  

## 4. Limitations
- May not reflect word-level understanding.  

## 5. Use Cases
- OCR evaluation.  
- ASR in character-based languages.  


# SPICE (Semantic Propositional Image Caption Evaluation)

## 1. What it is
**SPICE** is designed for image captioning.  
It parses candidate and reference captions into scene graphs (objects, attributes, relations) and compares them.

## 2. Formula
SPICE F1 = harmonic mean of precision and recall over semantic tuples.  

## 3. Advantages
- Captures semantic meaning, not just surface overlap.  
- Strong correlation with human judgment.  

## 4. Limitations
- Parsing errors can reduce accuracy.  
- Computationally heavier.  

## 5. Use Cases
- Image captioning benchmarks (COCO Captioning).  


# COMET

## 1. What it is
**COMET** is a neural evaluation metric for MT.  
It uses pretrained models (often multilingual transformers) fine-tuned on human ratings to predict quality.

## 2. Formula
Trained regression model:  
$$
Score = f_\theta(\text{source}, \text{hypothesis}, \text{reference})
$$

## 3. Advantages
- Strong correlation with human judgment.  
- Handles semantics and context.  

## 4. Limitations
- Requires pretrained/fine-tuned models.  
- Black-box nature.  

## 5. Use Cases
- Machine translation competitions (WMT).  
- Benchmarking modern MT systems.  


# QuestEval

## 1. What it is
**QuestEval** evaluates text generation (summarization, MT) by asking QA models to generate questions from reference text and checking if candidate answers them.

## 2. Advantages
- Evaluates factual consistency.  
- Works without exact lexical overlap.  

## 3. Limitations
- Depends on quality of QA system.  
- More complex pipeline.  

## 4. Use Cases
- Summarization evaluation.  
- Factual correctness in generated text.  


# BARTScore

## 1. What it is
**BARTScore** uses a pretrained sequence-to-sequence LM (BART) to compute likelihood of generating reference from candidate (and vice versa).  

## 2. Formula
$$
Score = \frac{1}{n} \sum_{i=1}^n \log P(y_i \,|\, x_{<i}; \theta)
$$

## 3. Advantages
- Leverages pretrained LMs.  
- Stronger correlation with human evaluation than BLEU/ROUGE.  

## 4. Limitations
- Model-dependent.  
- Computationally expensive.  

## 5. Use Cases
- Summarization and MT evaluation.  
- General NLG tasks.  


# Reinforcement Learning (RL) Metrics

Reinforcement Learning metrics evaluate how well an agent learns to maximize cumulative rewards through interactions with an environment.  
They measure **reward accumulation**, **policy efficiency**, and **learning stability**.  


# Cumulative Reward

## 1. What it is
**Cumulative Reward** is the total reward collected by an agent across a full episode or time horizon.

## 2. Formula
$$
R = \sum_{t=1}^T r_t
$$

Where \( r_t \) is the reward at step \( t \).  

## 3. Advantages
- Directly measures agent performance.  
- Easy to interpret.  

## 4. Limitations
- Depends heavily on reward function design.  
- Sensitive to episode length.  

## 5. Use Cases
- Benchmarking RL agents.  
- Comparing policies in games, robotics.  


# Average Reward per Episode

## 1. What it is
The **average reward per episode** normalizes cumulative reward by episode length, allowing fairer comparisons.

## 2. Formula
$$
\bar{R} = \frac{1}{N} \sum_{i=1}^N R_i
$$

Where \( R_i \) = cumulative reward for episode \( i \).  

## 3. Advantages
- Normalized comparison across episodes.  
- Reduces bias from episode length.  

## 4. Limitations
- Still dependent on reward scale.  

## 5. Use Cases
- Long vs. short horizon tasks.  
- Fair comparison of policies.  


# Discounted Return

## 1. What it is
**Discounted Return** evaluates cumulative reward with future rewards discounted by factor \( \gamma \).  
Encourages short-term gains while valuing long-term outcomes.

## 2. Formula
$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

Where \( 0 \leq \gamma \leq 1 \).  

## 3. Advantages
- Balances short- vs. long-term rewards.  
- Standard in RL training (e.g., Q-learning).  

## 4. Limitations
- Requires tuning \( \gamma \).  
- Interpretation depends on horizon length.  

## 5. Use Cases
- Most RL algorithms (Q-learning, policy gradients).  


# Success Rate

## 1. What it is
**Success Rate** measures the percentage of episodes where the agent achieves a predefined goal.

## 2. Formula
$$
SuccessRate = \frac{\text{Number of successful episodes}}{\text{Total episodes}}
$$

## 3. Advantages
- Simple and interpretable.  
- Useful for sparse-reward tasks.  

## 4. Limitations
- Ignores quality of performance beyond success.  
- Not informative for continuous tasks.  

## 5. Use Cases
- Robotics (reach target, pick-and-place).  
- Games (win vs. loss).  


# Regret

## 1. What it is
**Regret** measures the difference between the reward of the optimal policy and the learned policy.  
Lower regret = closer to optimal.

## 2. Formula
$$
Regret(T) = \sum_{t=1}^T (r^* - r_t)
$$

Where \( r^* \) = reward of optimal action at time \( t \).  

## 3. Advantages
- Standard in bandit/RL theory.  
- Quantifies inefficiency of learning.  

## 4. Limitations
- Requires knowledge of optimal policy (not always available).  

## 5. Use Cases
- Multi-armed bandits.  
- Theoretical RL performance guarantees.  


# Sample Efficiency

## 1. What it is
**Sample Efficiency** measures how quickly an RL agent learns a good policy relative to the number of interactions with the environment.  

## 2. Advantages
- Important in real-world tasks (robotics).  
- Differentiates practical algorithms from purely theoretical ones.  

## 3. Limitations
- No universal formula.  
- Depends on task definition.  

## 4. Use Cases
- Robotics, where training interactions are expensive.  
- Comparing RL algorithms.  


# Stability and Variance of Returns

## 1. What it is
Evaluates the consistency of returns across training episodes.  
Stable learning → low variance in performance.

## 2. Advantages
- Important for safe RL deployment.  
- Distinguishes reliable policies.  

## 3. Limitations
- Variance may mask true learning potential.  

## 4. Use Cases
- Real-world RL systems requiring robustness.  


# Generative Model Metrics

Generative model metrics evaluate how well models like **GANs, VAEs, and Diffusion Models** produce data that is both **realistic (fidelity)** and **diverse (coverage)**.  
Unlike standard ML tasks, evaluation here is challenging since the true data distribution is unknown.  
Metrics fall into categories:  
- **Distribution distance metrics** (FID, KID, Intra-FID)  
- **Quality/diversity trade-off metrics** (IS, Mode Score, PR-GAN, D&C)  
- **Perceptual similarity metrics** (LPIPS, SSIM)  
- **Older baseline methods** (KDE likelihood)  


# Inception Score (IS)

## 1. What it is
The **Inception Score** evaluates the quality and diversity of generated images using a pretrained classifier (Inception-v3).  
- High-quality images → confident predictions (low entropy per image).  
- Diverse images → uniform distribution across classes (high entropy overall).

## 2. Formula
$$
IS = \exp \left( \mathbb{E}_x \, KL(p(y|x) \| p(y)) \right)
$$

Where:  
- \( p(y|x) \) = predicted class distribution for generated image \( x \)  
- \( p(y) \) = marginal class distribution  

## 3. Interpretation
- Higher IS = better quality + diversity.  
- IS = 1 → trivial (all images same).  

## 4. Advantages
- Simple, widely adopted in early GAN research.  

## 5. Limitations
- Ignores realism relative to real data.  
- Biased by the classifier used (Inception).  

## 6. Use Cases
- Early GAN benchmarks.  
- Quick sanity checks for generative models.  


# Fréchet Inception Distance (FID)

## 1. What it is
**FID** compares the distribution of generated images to real images using Inception embeddings.  
It measures both fidelity and diversity.

## 2. Formula
$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

Where:  
- \( \mu_r, \Sigma_r \) = mean and covariance of real features  
- \( \mu_g, \Sigma_g \) = mean and covariance of generated features  

## 3. Interpretation
- Lower FID = closer to real data distribution.  

## 4. Advantages
- Captures both quality and diversity.  
- More robust than IS.  

## 5. Limitations
- Requires large sample sizes.  
- Sensitive to embedding network choice.  

## 6. Use Cases
- Standard GAN and diffusion benchmarks.  


# Kernel Inception Distance (KID)

## 1. What it is
**KID** is an alternative to FID that uses polynomial kernels in feature space.  
Unlike FID, it provides an **unbiased estimator**.

## 2. Formula
$$
KID = \text{MMD}^2(P_r, P_g)
$$

Where \( \text{MMD}^2 \) = squared Maximum Mean Discrepancy between real (P_r) and generated (P_g) distributions.  

## 3. Interpretation
- Lower KID = better model.  

## 4. Advantages
- More stable for small samples.  
- Unbiased compared to FID.  

## 5. Limitations
- Less widely adopted than FID.  

## 6. Use Cases
- GAN evaluation with limited data.  


# Mode Score

## 1. What it is
**Mode Score** extends IS by comparing generated label distribution to real label distribution, penalizing mode dropping.  

## 2. Formula
$$
ModeScore = \exp \left( \mathbb{E}_x KL(p(y|x) \| p(y)) - KL(p(y) \| p^*(y)) \right)
$$

Where:  
- \( p^*(y) \) = true distribution of real data labels  

## 3. Interpretation
- Higher Mode Score = high quality + diversity + real-data alignment.  

## 4. Advantages
- Improves on IS by considering real data.  

## 5. Limitations
- Still classifier-dependent.  
- Rarely used now (replaced by FID).  

## 6. Use Cases
- GANs with mode collapse issues.  


# Precision and Recall for Generative Models (PR-GAN)

## 1. What it is
**PR-GAN** evaluates generative models by separating:  
- **Precision** → sample quality (realistic images).  
- **Recall** → coverage of real data distribution.  

## 2. Formula
Computed via nearest-neighbor tests in embedding space:  
- Precision = fraction of generated samples close to real data.  
- Recall = fraction of real samples covered by generated distribution.  

## 3. Advantages
- Distinguishes between "few perfect samples" vs. "diverse but poor samples".  

## 4. Limitations
- Requires feature embedding choice.  
- Computationally heavy.  

## 5. Use Cases
- Evaluating GAN trade-offs (fidelity vs. diversity).  


# Density and Coverage (D&C)

## 1. What it is
**D&C** refines PR-GAN by computing:  
- **Density**: average number of real neighbors within a radius of generated samples.  
- **Coverage**: proportion of real samples covered by generated data.  

## 2. Advantages
- Strong correlation with human judgment.  
- Separates fidelity from diversity clearly.  

## 3. Limitations
- Sample-size dependent.  
- Embedding space choice matters.  

## 4. Use Cases
- Evaluating GANs and diffusion models.  


# Intra-FID / Class-Conditional FID

## 1. What it is
Variants of FID that evaluate per-class performance:  
- **Intra-FID**: FID computed within each class.  
- **Class-Conditional FID**: compares generated vs. real distributions class by class.  

## 2. Advantages
- Detects class imbalance or mode dropping.  

## 3. Limitations
- Requires labeled data.  
- More computationally expensive.  

## 4. Use Cases
- Conditional GAN evaluation.  
- Fine-grained class-level generation analysis.  


# Learned Perceptual Image Patch Similarity (LPIPS)

## 1. What it is
**LPIPS** uses deep network embeddings (e.g., AlexNet, VGG) to measure perceptual similarity between images.  

## 2. Formula
$$
LPIPS(x,y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (F_l(x)_{hw} - F_l(y)_{hw}) ||_2^2
$$

Where \( F_l(x) \) = feature maps at layer \( l \).  

## 3. Advantages
- Strong alignment with human perception.  
- Widely used in GAN and image restoration tasks.  

## 4. Limitations
- Requires pretrained networks.  
- More expensive than SSIM/PSNR.  

## 5. Use Cases
- GAN evaluation.  
- Super-resolution, style transfer, inpainting.  


# Kernel Density Estimate (KDE) Likelihood

## 1. What it is
An older method where generated data likelihood is estimated using kernel density on real samples.  

## 2. Formula
For bandwidth \( h \):
$$
p(x) = \frac{1}{nh^d} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)
$$

Where \( K \) = kernel function, \( d \) = data dimension.  

## 3. Advantages
- Simple and conceptually intuitive.  

## 4. Limitations
- Scales poorly with high dimensions.  
- Superseded by FID and PR-based metrics.  

## 5. Use Cases
- Historical GAN papers.  
- Toy datasets or low-dimensional benchmarks.  


#  Master Summary Table of ML Metrics

| Domain            | Metric(s)                                   | Type / Focus                       | Notes / Properties |
|-------------------|---------------------------------------------|------------------------------------|--------------------|
| **Regression**    | MAE, MedAE                                 | Scale-dependent error              | Robust (MedAE), interpretable |
|                   | MSE, RMSE                                  | Scale-dependent error              | RMSE penalizes outliers |
|                   | R², Adjusted R²                            | Goodness-of-fit                    | Explained variance |
|                   | MSLE                                       | Log-scale error                    | Handles growth rates |
|                   | MAPE, sMAPE, WAPE                          | Scale-free percentage error        | Sensitive to zero values |
|                   | Huber, Quantile Loss                       | Robust losses                      | Less sensitive to outliers |
| **Classification**| Accuracy                                   | Overall correctness                | Misleading on imbalance |
|                   | Precision, Recall, F1                      | Positive class performance         | Precision–Recall trade-off |
|                   | Specificity (TNR), Sensitivity (TPR)       | Neg/Pos detection                  | Complements Recall |
|                   | ROC-AUC, PR-AUC                            | Threshold-free discrimination      | PR-AUC better for imbalance |
|                   | Log-Loss, Cross-Entropy                    | Probabilistic correctness          | Penalizes miscalibration |
|                   | MCC, Cohen’s Kappa, Balanced Accuracy      | Imbalance-robust                   | Class agreement focus |
|                   | Brier Score, Hamming Loss, Jaccard Index   | Probabilistic / set-based          | Alternative evaluation |
| **Ranking / IR**  | Precision@k, Recall@k                      | Top-k relevance                    | Search, recommender systems |
|                   | MAP, NDCG, MRR                             | Ranking quality                    | IR & recommender benchmarks |
|                   | Hit Ratio, Coverage, Diversity, Novelty    | Recommendation systems             | Measures coverage/variety |
|                   | ERR, PRBEP, Mean Rank                      | Ranking relevance                  | Alternative measures |
| **Clustering**    | Silhouette, Davies–Bouldin, Calinski–Harabasz | Internal cluster validation     | Separation & cohesion |
|                   | Dunn Index                                 | Internal validity                  | Sensitive to noise |
|                   | ARI, NMI, Purity, Homogeneity, Completeness, V-Measure | External validation | Compare vs. ground truth |
|                   | Entropy, Gap Statistic                     | Cluster interpretability           | Gap = cluster number selection |
|                   | Modularity, AIC, BIC                       | Graph / model-based                | Community detection, model fit |
| **Time-Series**   | MAE, MedAE, MSE, RMSE                      | Scale-dependent error              | Standard baselines |
|                   | MSLE, RMSPE, RMSSE                         | Log / scale-adjusted error         | RMSSE used in M5 competition |
|                   | MAPE, sMAPE, WAPE, MAAPE                   | Scale-free percentage error        | MAAPE avoids division by zero |
|                   | MASE, Theil’s U                            | Relative to benchmark              | Robust across scales |
|                   | Bias (MFE)                                 | Forecast bias                      | Direction of error |
|                   | Pinball Loss                               | Quantile forecasting               | Used in probabilistic models |
|                   | PICP, Interval Score                       | Prediction interval quality        | Coverage & sharpness |
| **Computer Vision**| IoU, Dice, mIoU                           | Segmentation overlap               | Standard CV metrics |
|                   | Pixel Acc, mPA                             | Segmentation pixel accuracy        | Biased to majority classes |
|                   | Panoptic Quality (PQ)                      | Unified detection + segmentation   | Panoptic segmentation |
|                   | Boundary IoU / Boundary F1                 | Boundary quality                   | Medical imaging |
|                   | mAP                                        | Object detection                   | COCO, Pascal VOC |
|                   | PSNR, SSIM                                 | Image reconstruction quality       | PSNR = pixel, SSIM = structural |
|                   | LPIPS                                      | Perceptual similarity              | Deep feature-based |
|                   | FID, KID, Intra-FID                        | Generative distribution distance   | GAN/Diffusion benchmarks |
| **NLP**           | BLEU, ROUGE, METEOR, GLEU, ChrF            | Lexical overlap                    | MT, summarization |
|                   | TER, WER, CER                              | Edit distance                      | MT, ASR, OCR |
|                   | SPICE                                      | Semantic caption evaluation        | Image captioning |
|                   | COMET, QuestEval, BARTScore                | Neural/semantic evaluation         | Closer to human judgment |
|                   | BERTScore, MoverScore                      | Embedding similarity               | Contextualized evaluation |
|                   | Perplexity                                 | Language modeling quality          | Lower = better |
|                   | Exact Match (EM), F1 (QA)                  | QA-specific overlap                | Used in SQuAD, NQ |
| **Reinforcement Learning** | Cumulative Reward, Avg Reward, Discounted Return | Reward-based | Direct performance |
|                   | Success Rate                               | Goal completion                    | Robotics, games |
|                   | Regret                                     | Policy inefficiency                | RL theory, bandits |
|                   | Sample Efficiency                          | Learning speed                     | Critical in real-world RL |
|                   | Stability / Variance of Returns            | Policy robustness                  | Safe RL |
| **Generative Models**| Inception Score (IS), Mode Score        | Quality + diversity                | Early GAN benchmarks |
|                   | FID, KID, Intra-FID/Class-FID              | Distribution distance              | GAN/Diffusion standards |
|                   | PR-GAN (Precision & Recall), D&C           | Fidelity vs. diversity trade-off   | Human-aligned metrics |
|                   | LPIPS                                      | Perceptual similarity              | GANs, SR, style transfer |
|                   | KDE Likelihood                             | Older likelihood method            | Obsolete but historical |
