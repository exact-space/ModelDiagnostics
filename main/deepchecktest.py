import pandas as pd
import numpy as np
import json
import requests

# import deepchecks


from deepchecks.tabular import datasets

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity


import os
import datetime
import time
import statistics
import math
import sys

# import timeseries as ts
import os
import datetime
import time
import statistics
import math
import sys
import json
import os
import platform
import re
import time as t
from datetime import date, datetime, time, timedelta

import platform

version = platform.python_version().split(".")[0]
if version == "3":
    import app_config.app_config as cfg
elif version == "2":
    import app_config as cfg
config = cfg.getconfig()

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import logging as lg


global lag_inputs, forecast_steps
lag_inputs = 60
forecast_steps = 5


def getModelFromId(modelId):
    query = {"id": modelId}
    urlQuery = (
        config["api"]["meta"]
        + '/optimizationmodels?filter={"where":'
        + json.dumps(query)
        + "}"
    )
    response = requests.get(urlQuery)
    if response.status_code == 200:
        # print(response.status_code)
        print("Got model details successfully.....")
        modelDetails = json.loads(response.content)
    else:
        modelDetails = {}
        print("Did not get model details successfully.....")
        print(response.status_code)
        print(response.content)
    return modelDetails


def getTagsFromModelDetailes(modelDetails):
    inputTags = []
    outputTags = []

    for ip in modelDetails[0]["inputs"]:
        inputTags.append(ip["dataTagId"])
    for op in modelDetails[0]["outputs"]:
        outputTags.append(op["dataTagId"])

    modelDict = {
        "inputTags": inputTags,
        "outputTags": outputTags,
        "id": modelDetails[0]["id"],
    }

    return modelDict


def prepare_data2(lagDf, lag_inputs, forecast_steps, target_cols, feature_cols):
    print("Customizing data for training")
    # Convert the input dataframe to a numpy array
    data = lagDf.drop(target_cols, axis=1).values

    # Create arrays for the lagged inputs and target columns
    print(len(lagDf), lag_inputs, forecast_steps)
    input_data = np.zeros(
        (len(lagDf) - lag_inputs - forecast_steps + 1, lag_inputs * data.shape[1])
    )
    target_data = np.zeros(
        (
            len(lagDf) - lag_inputs - forecast_steps + 1,
            forecast_steps * len(target_cols),
        )
    )
    for i in range(lag_inputs, len(lagDf) - forecast_steps + 1):
        input_data[i - lag_inputs, :] = data[i - lag_inputs : i, :].flatten()
        target_data[i - lag_inputs, :] = lagDf.loc[
            lagDf.index[i : i + forecast_steps], target_cols
        ].values.flatten()

    # Combine the lagged inputs and target columns into a single numpy array
    new_data = np.concatenate((input_data, target_data), axis=1)

    # Create a new dataframe from the combined numpy array
    new_columns = [
        f"{column}_lag{i}"
        for i in range(1, lag_inputs + 1)
        for column in lagDf.columns
        if column not in target_cols
    ]
    inputColumns = [
        f"{column}_lag{i}" for i in range(1, lag_inputs + 1) for column in feature_cols
    ]

    outputColumns = []
    for i in range(1, forecast_steps + 1):
        for target_col in target_cols:
            outputColumns.append(f"{target_col}_t+{i}")

    new_columns = new_columns + outputColumns
    new_df = pd.DataFrame(new_data, columns=new_columns)

    return new_df, inputColumns, outputColumns


def getLatestVersionHistoryFromModelId(modelId):
    urlQuery = (
        config["api"]["meta"] + "/optimizationmodels/" + modelId + "/modelversions"
    )

    response = requests.get(urlQuery)

    if response.status_code == 200:
        versionHistory = json.loads(response.content)
        versionHistory = versionHistory[-1]
        currentVersion = versionHistory["version"]
    else:
        print("Getting latest version history failed....")
        versionHistory = {}
        currentVersion = 0
        print(response.status_code)
        print(response.content)
    return versionHistory, currentVersion


def downloadingFile(fileName):
    url = config["api"]["meta"] + "/attachments/models/download/" + fileName
    res = requests.get(url)
    open(fileName, "wb").write(res.content)
    print("Downloading completed for file " + str(fileName))


def downloadingHTMLfile(fileName):
    url = config["api"]["meta"] + "/attachments/modeldiagnostics/download/" + fileName
    res = requests.get(url)
    open(fileName, "wb").write(res.content)
    print("Downloading completed for file " + str(fileName))


def uploadTrainingResults(path, fileName):
    files = {"upload_file": open(str(path + fileName), "rb")}
    url = config["api"]["meta"] + "/attachments/modeldiagnostics/upload"

    response = requests.post(url, files=files)
    print("uploading")
    print(url)
    print("+" * 20)
    print("response", response)

    if response.status_code == 200:
        print(fileName + " uploaded successfully....")
        status = "success"
        print(path + fileName)
        # os.remove(str(path+fileName))
    else:
        print(fileName + " did not uploaded successfully....")
        status = str(response.status_code) + str(response.content)
    return status


def gettingTimeStamp(date):
    date = datetime.strptime(date, "%d-%m-%YT%H:%M:%S")
    timestamp = datetime.timestamp(date)
    return timestamp


def getValuesV2(tagList, startTime, endTime):
    url = config["api"]["query"]
    metrics = []
    for tag in tagList:
        tagDict = {
            "tags": {},
            "name": tag,
            "aggregators": [
                {
                    "name": "avg",
                    "sampling": {"value": "1", "unit": "minutes"},
                    "align_end_time": True,
                }
            ],
        }
        metrics.append(tagDict)

    query = {
        "metrics": metrics,
        "plugins": [],
        "cache_time": 0,
        "start_absolute": startTime,
        "end_absolute": endTime,
    }
    #     print(json.dumps(query,indent=4))
    res = requests.post(url=url, json=query)
    values = json.loads(res.content)
    finalDF = pd.DataFrame()
    for i in values["queries"]:
        df = pd.DataFrame(
            i["results"][0]["values"], columns=["time", i["results"][0]["name"]]
        )

        try:
            finalDF = pd.concat([finalDF, df.set_index("time")], axis=1)
        except Exception as e:
            print(e)
            finalDF = pd.concat([finalDF, df], axis=1)

    finalDF.reset_index(inplace=True)
    finalDF.fillna(method="ffill", inplace=True)
    finalDF.fillna(method="bfill", inplace=True)

    # print(dates)
    return finalDF


modelId = "641a9edc27caeb0007732d5a"  # first model

# modelId = "63a292cdf2a0f8000741e8aa"


modelDetails = getModelFromId(modelId)


# getting model details:
modelTags = getTagsFromModelDetailes(modelDetails)

inputTags = modelTags["inputTags"]
outputTags = modelTags["outputTags"]

totalTags = inputTags + outputTags

versionHist, currentVersion = getLatestVersionHistoryFromModelId(modelId)

# h5FileName = modelId + "_Version" + str(currentVersion) + "_ANN_Regression.h5"
# transFileName = modelId + "_Version" + str(currentVersion) + "_Transformer.pkl"

# downloadingFile(h5FileName)
# downloadingFile(transFileName)


refStartTime = versionHist["modelTime"]["train"]
refStartTime = refStartTime[0]["startTime"]
refStartTime = gettingTimeStamp(refStartTime) * 1000
refEndTime = versionHist["modelTime"]["train"]
refEndTime = refEndTime[0]["endTime"]
refEndTime = gettingTimeStamp(refEndTime) * 1000


# test
refTestTime = versionHist["modelTime"]["test"]
refTestTime = refTestTime[0]["startTime"]
refTestTime = gettingTimeStamp(refTestTime) * 1000
refTestEndTime = versionHist["modelTime"]["test"]
refTestEndTime = refTestEndTime[0]["endTime"]
refTestEndTime = gettingTimeStamp(refTestEndTime) * 1000


today = datetime.today()
v = datetime.combine(today, time.min)
l = v - timedelta(days=15)
currentStartTime = (
    int(t.mktime(l.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000
)
currenEndTime = int(t.mktime(v.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000

currentDataFrame = getValuesV2(totalTags, refStartTime, refEndTime).drop(
    ["time"], axis=1
)


# testDf = getValuesV2(totalTags, refTestTime, refTestEndTime).drop(["time"], axis=1)
# trainDf = getValuesV2(totalTags, refStartTime, refEndTime).drop(["time"], axis=1)

# X_train, inputColumnsTrain, outputColumnsTrain = prepare_data2(
#     trainDf, lag_inputs, forecast_steps, outputTags, inputTags
# )
# X_test, inputColumnsTrain, outputColumnsTrain = prepare_data2(
#     testDf, lag_inputs, forecast_steps, outputTags, inputTags
# )


# y_train = X_train[outputColumnsTrain]
# y_test = X_test[outputColumnsTrain]
# X_train = X_train[inputColumnsTrain]
# X_test = X_test[inputColumnsTrain]


# transformer = pickle.load(open(transFileName, "rb"))

# annModel = keras.models.load_model(h5FileName)


from deepchecks.tabular import Dataset

# dsTrain = Dataset(X_train, label=outputColumnsTrain)
# dsTest = Dataset(X_test, label= outputColumnsTrain)

# from deepchecks.tabular.suites import full_suite



ds = Dataset(currentDataFrame)

# Run Suite:
integ_suite = data_integrity()
suite_result = integ_suite.run(ds)



savingname = modelId + "_Version" + str(currentVersion)

suite_result.save_as_html(savingname + "data.html")

# # Note: the result can be saved as html using suite_result.save_as_html()
# # or exported to json using suite_result.to_json()
# suite_result.show()
# suite_result.save_as_html()


uploadTrainingResults(
    "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "data.html"
)
# uploadTrainingResults(
#     "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "model.html"
# )

# evaluation_suite = model_evaluation()
# suite_result = evaluation_suite.run(train_ds, test_ds, gbr)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()

# suite_result = suite.run(train_dataset=dsTrain, test_dataset=dsTest, model=annModel)
# suite_result.save_as_html(savingname + "model.html")


# uploadTrainingResults(
#     "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "model.html"
# )
# uploadTrainingResults(
#     "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "model.html"
# )
