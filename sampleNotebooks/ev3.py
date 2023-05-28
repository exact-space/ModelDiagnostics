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


import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

content = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
).content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(
        arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"]
    )


raw_data.index = raw_data.apply(
    lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
    axis=1,
)


target = "cnt"
prediction = "prediction"
numerical_features = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
categorical_features = [
    "season",
    "holiday",
    "workingday",
]  #'weathersit']


reference = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]
current = raw_data.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"]


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    reference[numerical_features + categorical_features],
    reference[target],
    test_size=0.3,
)


regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)


regressor.fit(X_train, y_train)


preds_train = regressor.predict(X_train)
preds_test = regressor.predict(X_test)


X_train["target"] = y_train
X_train["prediction"] = preds_train

X_test["target"] = y_test
X_test["prediction"] = preds_test


column_mapping = ColumnMapping()

column_mapping.target = "target"
column_mapping.prediction = "prediction"
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features


regression_perfomance_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
regression_perfomance_dashboard.calculate(
    X_train.sort_index(), X_test.sort_index(), column_mapping=column_mapping
)


regression_perfomance_dashboard.save("target_drift.html")


regressor.fit(reference[numerical_features + categorical_features], reference[target])

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
reference["prediction"] = ref_prediction

regression_perfomance_dashboard = Dashboard(
    tabs=[RegressionPerformanceTab(verbose_level=0)]
)
regression_perfomance_dashboard.calculate(
    reference, None, column_mapping=column_mapping
)
regression_perfomance_dashboard.save("prod.html")


current_prediction = regressor.predict(
    current[numerical_features + categorical_features]
)
current["prediction"] = current_prediction


regression_perfomance_dashboard.calculate(
    reference,
    current.loc["2011-01-29 00:00:00":"2011-02-07 23:00:00"],
    column_mapping=column_mapping,
)
regression_perfomance_dashboard.save("week1results.html")


RegressionPerformanceTab.list_widgets()


regression_perfomance_dashboard = Dashboard(
    tabs=[
        RegressionPerformanceTab(
            include_widgets=[
                "Regression Model Performance Report.",
                "Reference: Model Quality (+/- std)",
                "Current: Model Quality (+/- std)",
                "Current: Error (Predicted - Actual)",
                "Current: Error Distribution",
            ]
        )
    ]
)
regression_perfomance_dashboard.calculate(
    reference,
    current.loc["2011-02-07 00:00:00":"2011-02-14 23:00:00"],
    column_mapping=column_mapping,
)


regression_perfomance_dashboard.save("week2results.html")


regression_perfomance_dashboard.calculate(
    reference,
    current.loc["2011-02-15 00:00:00":"2011-02-21 23:00:00"],
    column_mapping=column_mapping,
)
regression_perfomance_dashboard.save("week3results.html")


column_mapping_drift = ColumnMapping()

column_mapping_drift.target = target
column_mapping_drift.prediction = prediction
column_mapping_drift.numerical_features = numerical_features
column_mapping_drift.categorical_features = []


data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(
    reference,
    current.loc["2011-02-14 00:00:00":"2011-02-21 23:00:00"],
    column_mapping_drift,
)

regression_perfomance_dashboard.save("week4results.html")


from evidently.calculations.stattests import StatTest


def _anderson_stat_test(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
):
    p_value = anderson_ksamp(np.array([reference_data, current_data]))[2]
    return p_value, p_value < threshold


anderson_stat_test = StatTest(
    name="anderson",
    display_name="Anderson test (p_value)",
    func=_anderson_stat_test,
    allowed_feature_types=["num"],
)

options = DataDriftOptions(
    feature_stattest_func=anderson_stat_test, nbinsx=20, confidence=0.90
)


the_dashboard = Dashboard(
    tabs=[
        RegressionPerformanceTab(
            include_widgets=[
                "Regression Model Performance Report.",
                "Reference: Model Quality (+/- std)",
                "Current: Model Quality (+/- std)",
                "Current: Error (Predicted - Actual)",
                "Current: Error Distribution",
            ]
        ),
        DataDriftTab(),
    ],
    options=[options],
)

the_dashboard.calculate(
    reference,
    current.loc["2011-02-14 00:00:00":"2011-02-21 23:00:00"],
    column_mapping_drift,
)

regression_perfomance_dashboard.save("weekoptimiseresults.html")
