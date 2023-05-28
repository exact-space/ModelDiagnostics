import pandas as pd
import numpy as np
import requests
import zipfile
import io

from datetime import datetime
from evidently.tests import *
from sklearn import datasets, ensemble
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)


import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)
from evidently.options import DataDriftOptions
from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    RegressionPerformanceProfileSection,
)


content = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
).content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(
        arc.open("hour.csv"),
        header=0,
        sep=",",
        parse_dates=["dteday"],
        index_col="dteday",
    )


# print(raw_data.head())

#training starts
target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']


reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']


regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)

regressor.fit(reference[numerical_features + categorical_features], reference[target])

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])

reference['prediction'] = ref_prediction
current['prediction'] = current_prediction


#Model Performance:
column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features



report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=reference, current_data=current)
report.save_html("reg.html")