import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

data = fetch_california_housing(as_frame=True)
housing_data = data.frame

housing_data.rename(columns={"MedHouseVal": "target"}, inplace=True)
housing_data["prediction"] = housing_data["target"].values + np.random.normal(
    0, 5, housing_data.shape[0]
)

reference = housing_data.sample(n=5000, replace=False)
"""Ref : Baseline model use to train.
The second dataset is the current production data.
"""
current = housing_data.sample(n=5000, replace=False)


"""Data Drift: This Preset compares the distributions of the model features and show which have drifted.
 When we do not have ground truth labels or actuals, evaluating input data drift can help understand
  if an ML model still operates in a familiar environment."""

report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

report.run(reference_data=reference, current_data=current)
report.save_html("test.html")
