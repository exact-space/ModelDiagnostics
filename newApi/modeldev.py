import requests
import json
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


url = 'https://data.exactspace.co/exactapi/boilerStressProfiles?filter={%22where%22:{%22unitsId%22:%226200712ad08cac6240360bbe%22,%22type%22:%22boilerTubeLeakPredictParameters_V3%22}}'
res = requests.get(url).json()

print(json.dumps(res, indent = 4))




loaded_svr_model = joblib.load('path_to_your_svr_model.pkl')


new_data = np.array([[feature1_value, feature2_value, ...],
                     [feature1_value, feature2_value, ...],
                     ...])


new_predictions = loaded_svr_model.predict(new_data)


actual_values = np.array([target1_value, target2_value, ...])
deviations = np.abs(new_predictions - actual_values)
average_deviation = np.mean(deviations)


explainer = shap.Explainer(loaded_svr_model, new_data)
shap_values = explainer(new_data)


shap.summary_plot(shap_values, new_data)


