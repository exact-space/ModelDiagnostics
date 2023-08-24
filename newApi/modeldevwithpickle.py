import time
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import pickle
import h5py

# Function to load the model from a pkl file
def load_model_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to load the model from an h5 file
def load_model_from_h5(file_path):
    model = None
    with h5py.File(file_path, 'r') as file:
        # Add your code to load the model from the h5 file here
        pass
    return model

# Function to train the model
def train_model(features):
    # Add your code to train the model here
    print("Model training...")
    print("Features:", features)


# Function to evaluate the model and save the result in JSON format
def evaluate_model(model, predictions, targets, features, filename):
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

# Specify the file paths for the pkl and h5 models
model_pkl_file = 'model.pkl'
model_h5_file = 'model.h5'

while True:
    # Wait for the specified interval
    time.sleep(check_interval)

    # Load the model from the pkl file
    model_pkl = load_model_from_pkl(model_pkl_file)

    # Generate random multivariate predictions, targets, and feature values for demonstration
    features = ['feature1', 'feature2', 'feature3']  # Dynamic list of features

    predictions = {}
    targets = {}
    for feature_name in features:
        predictions[feature_name] = np.random.rand()
        targets[feature_name] = np.random.rand()

    # Evaluate the model's deviation and save the result in JSON format
    deviation = evaluate_model(model_pkl, predictions, targets, features, 'evaluation_result.json')

    # Calculate the dynamic deviation threshold as 20% of the RMSE deviation
    dynamic_threshold = deviation_threshold * deviation['rmse']

    # Check if the model's deviation exceeds the dynamic threshold
    if deviation['rmse'] > dynamic_threshold:
        print("Model deviation exceeds threshold. Retraining the model...")
        train_model(features)

        # Calculate the average deviation for each feature across all instances
        average_deviations = {
            feature_name: np.mean(feature_deviation)
            for feature_name, feature_deviation in deviation['feature_deviations'].items()
        }

        # Sort the features based on their average deviation in descending order
        sorted_features = sorted(average_deviations, key=average_deviations.get, reverse=True)

        # Print the features and their average deviations
        print("Feature Dominance:")
        for feature_name in sorted_features:
            print(f"{feature_name}: {average_deviations[feature_name]}")



