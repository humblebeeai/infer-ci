#!/usr/bin/env python3
"""
Demo script showing how to use the unified MetricEvaluator interface
"""
import sys
sys.path.append('/home/zokirov_diyorbek/Documents/confidence_interval_jacob/confidenceinterval')

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import the unified interface
from confidenceinterval import MetricEvaluator

def demo_regression():
    """Demonstrate regression evaluation with confidence intervals."""
    print("🔢 REGRESSION EVALUATION DEMO")
    print("=" * 50)
    
    # Generate sample regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Initialize evaluator
    evaluator = MetricEvaluator()
    
    print("\n📊 Available regression metrics:", evaluator.get_available_metrics('regression'))
    print("🛠️ Available methods:", evaluator.get_available_methods('regression'))
    
    # Test different metrics with different methods
    metrics_to_test = ['mae', 'mse', 'rmse', 'r2']
    methods_to_test = ['bootstrap_bca', 'jackknife']
    
    for metric in metrics_to_test:
        print(f"\n📈 Testing {metric.upper()}:")
        for method in methods_to_test:
            try:
                score, ci = evaluator.evaluate(
                    y_true=y_test.tolist(),
                    y_pred=y_pred.tolist(),
                    task='regression',
                    metric=metric,
                    method=method,
                    confidence_level=0.95
                )
                print(f"  {method:15s}: {score:.4f}, CI: ({ci[0]:.4f}, {ci[1]:.4f})")
            except Exception as e:
                print(f"  {method:15s}: Failed - {e}")

def demo_classification():
    """Demonstrate classification evaluation with confidence intervals."""
    print("\n\n🎯 CLASSIFICATION EVALUATION DEMO") 
    print("=" * 50)
    
    # Generate sample classification data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Initialize evaluator
    evaluator = MetricEvaluator()
    
    print("\n📊 Available classification metrics:", evaluator.get_available_metrics('classification'))
    print("🛠️ Available methods:", evaluator.get_available_methods('classification'))
    
    # Test different metrics
    metrics_to_test = ['accuracy', 'precision', 'recall']
    
    for metric in metrics_to_test:
        print(f"\n📈 Testing {metric}:")
        try:
            score, ci = evaluator.evaluate(
                y_true=y_test.tolist(),
                y_pred=y_pred.tolist(),
                task='classification',
                metric=metric,
                method='wilson',  # Default for classification
                confidence_level=0.95
            )
            print(f"  wilson         : {score:.4f}, CI: ({ci[0]:.4f}, {ci[1]:.4f})")
        except Exception as e:
            print(f"  wilson         : Failed - {e}")

def demo_simple_usage():
    """Show the simplest possible usage."""
    print("\n\n✨ SIMPLE USAGE EXAMPLES")
    print("=" * 50)
    
    # Simple regression example
    y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_reg = [1.1, 2.1, 2.9, 4.1, 4.9]
    
    evaluator = MetricEvaluator()
    mae_score, mae_ci = evaluator.evaluate(y_true_reg, y_pred_reg, task='regression', metric='mae')
    print(f"📊 Regression MAE: {mae_score:.4f}, CI: ({mae_ci[0]:.4f}, {mae_ci[1]:.4f})")
    
    # Simple classification example  
    y_true_clf = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred_clf = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    
    acc_score, acc_ci = evaluator.evaluate(y_true_clf, y_pred_clf, task='classification', metric='accuracy')
    print(f"🎯 Classification Accuracy: {acc_score:.4f}, CI: ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})")

if __name__ == "__main__":
    print("🚀 CONFIDENCE INTERVAL METRICS - UNIFIED INTERFACE DEMO")
    print("=" * 60)
    
    try:
        demo_regression()
        demo_classification() 
        demo_simple_usage()
        
        print("\n\n🎉 All demos completed successfully!")
        print("💡 You can now use MetricEvaluator for all your evaluation needs!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
