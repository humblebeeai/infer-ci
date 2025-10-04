"""
Regression Confidence Interval Validation Test

This test replicates the functionality from example_reg.ipynb to validate
confidence intervals for regression metrics using synthetic datasets.

Tests both Linear Regression and XGBoost models across multiple metrics:
- R² (Coefficient of Determination)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

The test validates that confidence intervals computed from test data contain
the actual metric values computed on validation data.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the main module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

from confidenceinterval import MetricEvaluator 

def load_california_housing_data():
    """Load and prepare California housing dataset."""
    print("Loading California housing dataset...")
    
    # Load California housing data
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print(f"Dataset shape: {df.shape}")
    print(f"Null values: {df.isnull().sum().sum()}")
    
    # Get features and target
    feature_cols = [col for col in df.columns if col != 'MedHouseVal']
    x = df[feature_cols]
    y = df['MedHouseVal']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return x, y


def test_regression_ci_single_experiment(x, y, dataset_name="Dataset"):
    """
    Test confidence intervals for regression metrics with a single experiment.
    
    This function replicates the single experiment logic from example_reg.ipynb
    """
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Initialize evaluator
    evaluate = MetricEvaluator()
    
    # Configuration
    metrics = ['r2', 'mse', 'mae', 'rmse']
    random_state = 42
    
    print(f"Testing metrics: {metrics}")
    print(f"Random state: {random_state}")
    
    # Data splitting: 50% train, 10% test, 40% validation
    x_train_single, x_temp_single, y_train_single, y_temp_single = train_test_split(
        x, y, test_size=0.5, random_state=random_state
    )
    x_test_single, x_val_single, y_test_single, y_val_single = train_test_split(
        x_temp_single, y_temp_single, test_size=0.8, random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(x_train_single)} samples")
    print(f"  Test: {len(x_test_single)} samples")
    print(f"  Validation: {len(x_val_single)} samples")
    
    # Feature scaling
    scaler_single = StandardScaler()
    x_train_single_scaled = scaler_single.fit_transform(x_train_single)
    x_test_single_scaled = scaler_single.transform(x_test_single)
    x_val_single_scaled = scaler_single.transform(x_val_single)
    
    # Train Linear Regression
    print(f"\nTraining Linear Regression...")
    lr_single = LinearRegression()
    lr_single.fit(x_train_single_scaled, y_train_single)
    y_test_pred_lr_single = lr_single.predict(x_test_single_scaled)
    y_val_pred_lr_single = lr_single.predict(x_val_single_scaled)
    
    # Train XGBoost
    print(f"Training XGBoost...")
    xgb_single = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_state,
        verbosity=0  # Suppress XGBoost output
    )
    xgb_single.fit(x_train_single, y_train_single)
    y_test_pred_xgb_single = xgb_single.predict(x_test_single)
    y_val_pred_xgb_single = xgb_single.predict(x_val_single)
    
    print("Models trained successfully!")
    
    # Evaluate each metric for both models
    results = {}
    all_coverage_success = True
    
    print(f"\n{'='*50}")
    print("CONFIDENCE INTERVAL VALIDATION RESULTS")
    print('='*50)
    
    for metric in metrics:
        print(f"\n=== {metric.upper()} METRIC ===")
        
        # Linear Regression evaluation
        lr_result = evaluate.evaluate(
            y_true=y_test_single, 
            y_pred=y_test_pred_lr_single, 
            task='regression', 
            metric=metric
        )
        lr_test_value, (lr_ci_lower, lr_ci_upper) = lr_result
        
        # Compute validation metric value
        if metric == 'r2':
            lr_val_actual = evaluate.r2_score(y_val_single, y_val_pred_lr_single, compute_ci=False)
        elif metric == 'mse':
            lr_val_actual = evaluate.mse(y_val_single, y_val_pred_lr_single, compute_ci=False)
        elif metric == 'mae':
            lr_val_actual = evaluate.mae(y_val_single, y_val_pred_lr_single, compute_ci=False)
        elif metric == 'rmse':
            lr_val_actual = evaluate.rmse(y_val_single, y_val_pred_lr_single, compute_ci=False)
        
        lr_coverage = lr_ci_lower <= lr_val_actual <= lr_ci_upper
        
        print(f"Linear Regression:")
        print(f"  Test {metric}: {lr_test_value:.4f}")
        print(f"  Bootstrap CI: [{lr_ci_lower:.4f}, {lr_ci_upper:.4f}]")
        print(f"  Validation {metric}: {lr_val_actual:.4f}")
        print(f"  CI Contains Validation: {lr_coverage}")
        
        # XGBoost evaluation
        xgb_result = evaluate.evaluate(
            y_true=y_test_single, 
            y_pred=y_test_pred_xgb_single, 
            task='regression', 
            metric=metric
        )
        xgb_test_value, (xgb_ci_lower, xgb_ci_upper) = xgb_result
        
        # Compute validation metric value
        if metric == 'r2':
            xgb_val_actual = evaluate.r2_score(y_val_single, y_val_pred_xgb_single, compute_ci=False)
        elif metric == 'mse':
            xgb_val_actual = evaluate.mse(y_val_single, y_val_pred_xgb_single, compute_ci=False)
        elif metric == 'mae':
            xgb_val_actual = evaluate.mae(y_val_single, y_val_pred_xgb_single, compute_ci=False)
        elif metric == 'rmse':
            xgb_val_actual = evaluate.rmse(y_val_single, y_val_pred_xgb_single, compute_ci=False)
        
        xgb_coverage = xgb_ci_lower <= xgb_val_actual <= xgb_ci_upper
        
        print(f"XGBoost:")
        print(f"  Test {metric}: {xgb_test_value:.4f}")
        print(f"  Bootstrap CI: [{xgb_ci_lower:.4f}, {xgb_ci_upper:.4f}]")
        print(f"  Validation {metric}: {xgb_val_actual:.4f}")
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
    print(f"\n{'='*40}")
    print("SUMMARY RESULTS")
    print('='*40)
    
    success_count = 0
    total_count = len(metrics) * 2  # 2 models per metric
    
    for metric in metrics:
        lr_coverage = results[metric]['lr_coverage']
        xgb_coverage = results[metric]['xgb_coverage']
        lr_width = results[metric]['lr_ci_width']
        xgb_width = results[metric]['xgb_ci_width']
        
        print(f"{metric.upper()}:")
        print(f"  Linear Regression: {'✓' if lr_coverage else '✗'} (CI width: {lr_width:.4f})")
        print(f"  XGBoost: {'✓' if xgb_coverage else '✗'} (CI width: {xgb_width:.4f})")
        
        if lr_coverage:
            success_count += 1
        if xgb_coverage:
            success_count += 1
    
    success_rate = (success_count / total_count) * 100
    print(f"\nOverall Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    return all_coverage_success, results


if __name__ == "__main__":
    print("Regression Confidence Interval Test")
    print("=" * 80)
    
    try:

        
        # California Housing dataset
        print(f"\n CALIFORNIA HOUSING DATASET TEST")
        try:
            x_housing, y_housing = load_california_housing_data()
            
            # Single experiment test
            housing_success, _ = test_regression_ci_single_experiment(
                x_housing, y_housing, "California Housing"
            )
                
        except Exception as e:
            print(f" California Housing test failed: {str(e)}")
            housing_success = False
        
        print(f"\n{'='*80}")
        if housing_success:
            print(" REGRESSION CI TESTS COMPLETED SUCCESSFULLY!")
        else:
            print(" REGRESSION CI TESTS COMPLETED WITH SOME ISSUES")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()