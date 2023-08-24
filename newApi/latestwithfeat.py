import time
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# Function to train the model
def train_model(features):
    # Add your code to train the model here
    print("Model training...")
    print("Features:", features)

# Function to evaluate the model and save the result in JSON format
def evaluate_model(predictions, targets, features, filename):
    # Extract the values from dictionaries and convert them to lists
    predictions_list = [predictions[feature] for feature in features]
    targets_list = [targets[feature] for feature in features]

    mse = mean_squared_error(targets_list, predictions_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_list, predictions_list)
    r2 = r2_score(targets_list, predictions_list)

    evaluation_result = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_deviations': {}
    }

    # Calculate feature deviations
    for feature_name in features:
        feature_deviation = np.abs(predictions[feature_name] - targets[feature_name])
        evaluation_result['feature_deviations'][feature_name] = feature_deviation.tolist()

    with open(filename, 'w') as json_file:
        json.dump(evaluation_result, json_file, indent=4)

    # Create a bar plot using Plotly for evaluation metrics
    metrics = list(evaluation_result.keys())[:-1]  # Exclude feature_deviations
    values = list(evaluation_result.values())[:-1]  # Exclude feature_deviations

    fig = go.Figure(data=[go.Bar(x=metrics, y=values)])

    fig.update_layout(
        title="Model Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Values"
    )

    fig.show()

    # Create a bar plot using Plotly for feature deviations
    fig2 = go.Figure()

    for feature_name, feature_deviation in evaluation_result['feature_deviations'].items():
        fig2.add_trace(go.Bar(x=[feature_name], y=[feature_deviation]))

    fig2.update_layout(
        title="Feature Deviations",
        xaxis_title="Features",
        yaxis_title="Deviations"
    )

    fig2.show()

    return evaluation_result

# Set the time interval for checking deviations (in seconds)
check_interval = 15  

# Define the initial features
features = ['feature1', 'feature2', 'feature3']

# Train the initial model
train_model(features)

# Initialize the dynamic deviation threshold
deviation_threshold = 0.1

while True:
    
    time.sleep(check_interval)
    features = ['feature1', 'feature2', 'feature3']  

    predictions = {}
    targets = {}
    for feature_name in features:
        predictions[feature_name] = np.random.rand()
        targets[feature_name] = np.random.rand()

 
    deviation = evaluate_model(predictions, targets, features, 'evaluation_result.json')

    
    dynamic_threshold = deviation['rmse'] * 0.2

    
    if deviation['rmse'] > dynamic_threshold:
        print("Model deviation exceeds threshold. Retraining the model...")
        train_model(features)

        
        print("Feature Deviations:")
        for feature_name, feature_deviation in deviation['feature_deviations'].items():
            print(f"{feature_name}: {feature_deviation}")
           
