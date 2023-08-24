import time
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go


def train_model():
    
    print("Model training...")

    predData = something.pkl(input)
    realData = realFromES


# Function to evaluate the model and save the result in JSON format
def evaluate_model(predictions, targets, filename):
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    evaluation_result = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    with open(filename, 'w') as json_file:
        json.dump(evaluation_result, json_file, indent=4)

    # Create a bar plot using Plotly
    metrics = list(evaluation_result.keys())
    values = list(evaluation_result.values())

    fig = go.Figure(data=[go.Bar(x=metrics, y=values)])

    fig.update_layout(
        title="Model Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Values"
    )

    fig.show()

    return evaluation_result

# Set the time interval for checking deviations (in seconds)
check_interval = 10  # Check every hour  

# Train the initial model
train_model()

# Initialize the initial threshold value
threshold = 0.1

while True:
    # Wait for the specified interval
    time.sleep(check_interval)

    
    predictions = np.random.rand(5)
    targets = np.random.rand(5)
    # Evaluate the model's deviation and save the result in JSON format
    deviation = evaluate_model(predictions, targets, 'evaluation_result.json')


    dynamic_threshold = deviation['rmse'] * 0.2  

    # Check if the model's deviation exceeds the dynamic threshold
    if deviation['rmse'] > dynamic_threshold:
        print("Model deviation exceeds threshold. Retraining the model...")#
        train_model()#call the function to train the model
    # Update the threshold with the dynamic threshold for the next iteration
    threshold = dynamic_threshold 

