import pandas as pd
from confidenceinterval import MetricEvaluator

df = pd.read_csv('/home/diyorbek/Downloads/Galaxy_S22_mironshoh_aggregated_results.csv')

print(f"The size of the dataframe: {df.shape}")
print(df.info())

print(f"This is the whole dataset: {df.head()}")

df = df.drop(columns=['Unnamed: 11'])
print(df.head())

df1 = df[['ground_truth', 'prediction1']]
print(f"This is 1st dataframe: {df1.head()}")

df2 = df[['ground_truth', 'prediction2']]
print(f"This is 2nd dataframe: {df2.head()}")

df3 = df[['ground_truth', 'prediction3']]
print(f"This is 3rd dataframe: {df3.head()}")

df4 = df[['ground_truth', 'prediction4']]
print(f"This is 4th dataframe: {df4.head()}")

df5 = df[['ground_truth', 'prediction5']]
print(f"This is 5th dataframe: {df5.head()}")

evaluate = MetricEvaluator()

dataframes = [df1, df2, df3, df4, df5]
predictions = ['prediction1', 'prediction2', 'prediction3', 'prediction4', 'prediction5']
results = {}

for i, dataframe in enumerate(dataframes):
    print(f"\nEvaluating dataframe {i+1}")
    result = evaluate.evaluate(y_true=dataframe['ground_truth'], y_pred=dataframe[predictions[i]], task='regression', metric='mape')
    
    # Extract values from the tuple: (metric_value, (lower_bound, upper_bound))
    metric_value, (lower_bound, upper_bound) = result
    
    # Store results
    results[f'dataset_{i+1}'] = {
        'score': metric_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    # Print results with confidence intervals
    print(f"Dataset {i+1} Results:")
    print(f"  MAPE Score: {metric_value:.4f}")
    print(f"  Lower Bound: {lower_bound:.4f}")
    print(f"  Upper Bound: {upper_bound:.4f}")
    print(f"  Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Print summary of all results
for dataset, result in results.items():
    print(f"{dataset}: MAPE = {result['score']:.4f} [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")
for dataset, result in results.items():
    print(f"{dataset}: MAPE = {result['score']:.4f} [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")

    