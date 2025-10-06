from confidenceinterval import MetricEvaluator
import pandas as pd
import xgboost as xgb

# Load the dataset
df = pd.read_csv("datasets/IV-Drip.csv")

# Initialize MetricEvaluator
evaluate = MetricEvaluator()

# Prepare the true and predicted values
y_true = []
y_pred = []
for i in range(len(df)):
    true = df.iloc[i]["ground_truth"]
    for j in range(1, 6):
        pred = df.iloc[i][f"prediction{j}"]
        if not pd.isna(pred):
            y_true.append(true)
            y_pred.append(pred)

# Initialize the evaluator
evaluator = MetricEvaluator()

# Evaluate the model with MAPE
mape, ci = evaluator.evaluate(y_true, y_pred, task="regression", metric="mape", plot=True)

# Format output for MAPE and Confidence Interval (CI)
ci_lower, ci_upper = ci  # Extract CI bounds
accuracy = 100 - mape  # Accuracy calculation based on MAPE

# Print results in a clear format
print(f"MAPE: {mape:.2f}%")
print(f"Confidence Interval(MAPE): [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Confidence Interval (Accuracy): [{100 - ci_upper:.2f}%, {100 - ci_lower:.2f}%]")
