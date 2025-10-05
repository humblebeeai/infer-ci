import pandas as pd
import numpy as np
from confidenceinterval import MetricEvaluator
import warnings 
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb 


def test_pipeline(path, ground_truth):

    """
    This function tests the regression pipeline on a given dataset.
    It includes data loading, exploratory data analysis, model training with Linear Regression and XGBoost,
    and evaluation of regression metrics with confidence intervals.

    Parameters:
    - path: str, the file path to the dataset
    - ground_truth: str, the name of the ground truth column

    Returns:
    - results: dict, containing coverage and CI width for each metric and model     
    """
    # Load dataset
    df = pd.read_csv(path)

    # Perform exploratory data analysis
    print(df.head())
    print(f"The size of the dataframe: {df.shape}")
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())


    # Get features and target
    # Exclude non-numeric columns (categorical features) and target column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != ground_truth]
    x = df[feature_cols]
    y = df[ground_truth]

    # Handle missing values by filling with median
    x = x.fillna(x.median())
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Missing values after imputation: {x.isnull().sum().sum()}")

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

    results = {}

    # Evaluate all metrics at once for Linear Regression
    lr_results = evaluate.evaluate_multiple(
        y_true=y_test_single,
        y_pred=y_test_pred_lr_single,
        task='regression',
        metrics=metrics,
        method='bootstrap_bca'
    )
    
    # Evaluate all metrics at once for XGBoost
    xgb_results = evaluate.evaluate_multiple(
        y_true=y_test_single,
        y_pred=y_test_pred_xgb_single,
        task='regression',
        metrics=metrics,
        method='bootstrap_bca'
    )
    
    # Evaluate all metrics for Linear Regression validation data (without CI)
    lr_val_results = evaluate.evaluate_multiple(
        y_true=y_val_single,
        y_pred=y_val_pred_lr_single,
        task='regression',
        metrics=metrics,
        compute_ci=False
    )
    
    # Evaluate all metrics for XGBoost validation data (without CI)
    xgb_val_results = evaluate.evaluate_multiple(
        y_true=y_val_single,
        y_pred=y_val_pred_xgb_single,
        task='regression',
        metrics=metrics,
        compute_ci=False
    )
    
    for metric in metrics:
        
        # Extract Linear Regression results
        lr_test_value, (lr_ci_lower, lr_ci_upper) = lr_results[metric]
        lr_val_actual = lr_val_results[metric]
        lr_coverage = lr_ci_lower <= lr_val_actual <= lr_ci_upper
        
        print(f"Linear Regression:")
        print(f"  Test {metric}: {lr_test_value:.4f}")
        print(f"  Bootstrap CI: [{lr_ci_lower:.4f}, {lr_ci_upper:.4f}]")
        print(f"  Validation {metric}: {lr_val_actual:.4f}")
        print(f"  CI Contains Validation: {lr_coverage}")
        
        # Extract XGBoost results
        xgb_test_value, (xgb_ci_lower, xgb_ci_upper) = xgb_results[metric]
        xgb_val_actual = xgb_val_results[metric]
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
        

    return results
if __name__ == "__main__":
    path = "/home/diyorbek/Downloads/housing.csv"
    ground_truth = "median_house_value"
    test_pipeline(path, ground_truth)

    