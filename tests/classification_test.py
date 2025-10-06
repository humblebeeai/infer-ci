import pandas as pd
import numpy as np
from confidenceinterval import MetricEvaluator
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb 


def load_data(path, ground_truth):
    """
    This function tests the binary classification pipeline on a given dataset.
    It includes data loading, model training with Logistic Regression and XGBoost,
    and evaluation of binary classification metrics with confidence intervals.

    Parameters:
    - path: str, the file path to the dataset
    - ground_truth: str, the name of the ground truth column

    Returns:
    - results: dict, containing coverage and CI width for each metric and model     
    """
    # Load dataset
    df = pd.read_csv(path)

    # Get features and target
    print(f"Dataset shape: {df.shape}")
    print(f"Ground truth column: {ground_truth}")

    corr_matrix = df.corr()
    high_corr_features = corr_matrix.index[abs(corr_matrix[ground_truth]) > 0.5].to_list()
    high_corr_features.remove(ground_truth)

    print(f"Selected {len(high_corr_features)} highly correlated features")
    print(f"Features: {high_corr_features}")

    y = df[ground_truth]
    x = df[high_corr_features]
    
    print(f"Class distribution:")
    print(y.value_counts())

    return x, y
    
def test_binary_classification_ci(path, ground_truth):
    """
    Test confidence intervals for binary classification metrics.
    
    This function replicates the main evaluation logic from example_binary.ipynb
    """
    
    # Load and prepare data
    x, y = load_data(path, ground_truth)
    
    # Initialize evaluator
    evaluate = MetricEvaluator()
    
    # Configuration
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    random_state = 42
    
    print(f"\nTesting metrics: {metrics}")
    print(f"Random state: {random_state}")
    
    # Data splitting
    x_train_exp, x_temp, y_train_exp, y_temp = train_test_split(
        x, y, test_size=0.5, random_state=random_state
    )
    x_test_exp, x_val_exp, y_test_exp, y_val_exp = train_test_split(
        x_temp, y_temp, test_size=0.8, random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(x_train_exp)} samples")
    print(f"  Test: {len(x_test_exp)} samples") 
    print(f"  Validation: {len(x_val_exp)} samples")
    
    # Feature scaling for logistic regression
    scaler_exp = StandardScaler()
    x_train_scaled = scaler_exp.fit_transform(x_train_exp)
    x_test_scaled = scaler_exp.transform(x_test_exp)
    x_val_scaled = scaler_exp.transform(x_val_exp)
    
    # Train Logistic Regression
    print(f"\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(x_train_scaled, y_train_exp)
    y_test_pred_lr = lr_model.predict(x_test_scaled)
    y_val_pred_lr = lr_model.predict(x_val_scaled)
    
    # Train XGBoost
    print(f"Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_state,
        verbosity=0  # Suppress XGBoost output
    )
    xgb_model.fit(x_train_exp, y_train_exp)
    y_test_pred_xgb = xgb_model.predict(x_test_exp)
    y_val_pred_xgb = xgb_model.predict(x_val_exp)
    
    
    # Evaluate each metric for both models
    results = {}
    all_coverage_success = True
    
    
    for metric in metrics:
        
        # Logistic Regression evaluation
        lr_result = evaluate.evaluate(
            y_true=y_test_exp, 
            y_pred=y_test_pred_lr, 
            task='classification', 
            metric=metric,
            plot=True
        )
        lr_test_value, (lr_ci_lower, lr_ci_upper) = lr_result
        
        # Compute validation metric value
        if metric == 'accuracy':
            lr_val_value = evaluate.accuracy_score(y_val_exp, y_val_pred_lr, compute_ci=False)
        elif metric == 'precision':
            lr_val_value = evaluate.ppv_score(y_val_exp, y_val_pred_lr, compute_ci=False)
        elif metric == 'recall':
            lr_val_value = evaluate.tpr_score(y_val_exp, y_val_pred_lr, compute_ci=False)
        elif metric == 'f1':
            precision = evaluate.ppv_score(y_val_exp, y_val_pred_lr, compute_ci=False)
            recall = evaluate.tpr_score(y_val_exp, y_val_pred_lr, compute_ci=False)
            lr_val_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        lr_coverage = lr_ci_lower <= lr_val_value <= lr_ci_upper
        
        print(f"Logistic Regression:")
        print(f"  Test {metric}: {lr_test_value:.4f}")
        print(f"  Bootstrap CI: [{lr_ci_lower:.4f}, {lr_ci_upper:.4f}]")
        print(f"  Validation {metric}: {lr_val_value:.4f}")
        print(f"  CI Contains Validation: {lr_coverage}")
        
        # XGBoost evaluation
        xgb_result = evaluate.evaluate(
            y_true=y_test_exp, 
            y_pred=y_test_pred_xgb, 
            task='classification', 
            metric=metric
        )
        xgb_test_value, (xgb_ci_lower, xgb_ci_upper) = xgb_result
        
        # Compute validation metric value
        if metric == 'accuracy':
            xgb_val_value = evaluate.accuracy_score(y_val_exp, y_val_pred_xgb, compute_ci=False)
        elif metric == 'precision':
            xgb_val_value = evaluate.ppv_score(y_val_exp, y_val_pred_xgb, compute_ci=False)
        elif metric == 'recall':
            xgb_val_value = evaluate.tpr_score(y_val_exp, y_val_pred_xgb, compute_ci=False)
        elif metric == 'f1':
            precision = evaluate.ppv_score(y_val_exp, y_val_pred_xgb, compute_ci=False)
            recall = evaluate.tpr_score(y_val_exp, y_val_pred_xgb, compute_ci=False)
            xgb_val_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        xgb_coverage = xgb_ci_lower <= xgb_val_value <= xgb_ci_upper
        
        print(f"XGBoost:")
        print(f"  Test {metric}: {xgb_test_value:.4f}")
        print(f"  Bootstrap CI: [{xgb_ci_lower:.4f}, {xgb_ci_upper:.4f}]")
        print(f"  Validation {metric}: {xgb_val_value:.4f}")
        print(f"  CI Contains Validation: {xgb_coverage}")
        
        # Store results
        results[metric] = {
            'lr_coverage': lr_coverage,
            'xgb_coverage': xgb_coverage,
            'lr_ci_width': lr_ci_upper - lr_ci_lower,
            'xgb_ci_width': xgb_ci_upper - xgb_ci_lower
        }
        
        # Track overall success
        if not (lr_coverage and xgb_coverage):
            all_coverage_success = False
    
    # Print summary
    
    success_count = 0
    total_count = len(metrics) * 2  # 2 models per metric
    
    for metric in metrics:
        lr_coverage = results[metric]['lr_coverage']
        xgb_coverage = results[metric]['xgb_coverage']
        lr_width = results[metric]['lr_ci_width']
        xgb_width = results[metric]['xgb_ci_width']
        
        print(f"{metric.upper()}:")
        print(f"  Logistic Regression: {'✓' if lr_coverage else '✗'} (CI width: {lr_width:.4f})")
        print(f"  XGBoost: {'✓' if xgb_coverage else '✗'} (CI width: {xgb_width:.4f})")
        
        if lr_coverage:
            success_count += 1
        if xgb_coverage:
            success_count += 1
    
    success_rate = (success_count / total_count) * 100
    print(f"\nOverall Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")

if __name__ == "__main__":
    path = 'datasets/breast_cancer.csv'
    ground_truth = 'target'
    test_passed = test_binary_classification_ci(path, ground_truth)