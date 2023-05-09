import warnings

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics

from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.metric_preset import (
    TargetDriftPreset,
    RegressionPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.tests import *

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import json
import requests

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


modelId = "63a0b29d8c8c0600070c22e0"

modelDetails = getModelFromId(modelId)


# getting model details:
modelTags = getTagsFromModelDetailes(modelDetails)

inputTags = modelTags["inputTags"]
outputTags = modelTags["outputTags"]


totalTags = inputTags + outputTags


versionHist, currentVersion = getLatestVersionHistoryFromModelId(modelId)

h5FileName = modelId + "_Version" + str(currentVersion) + "_ANN_Regression.h5"
transFileName = modelId + "_Version" + str(currentVersion) + "_Transformer.pkl"

downloadingFile(h5FileName)
downloadingFile(transFileName)


refStartTime = versionHist["modelTime"]["train"]
refStartTime = refStartTime[0]["startTime"]
refStartTime = gettingTimeStamp(refStartTime) * 1000
refEndTime = versionHist["modelTime"]["train"]
refEndTime = refEndTime[0]["endTime"]
refEndTime = gettingTimeStamp(refEndTime) * 1000


today = datetime.today()
v = datetime.combine(today, time.min)
l = v - timedelta(days=15)
currentStartTime = (
    int(t.mktime(l.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000
)
currenEndTime = int(t.mktime(v.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000

refDataFrame = getValuesV2(totalTags, refStartTime, refEndTime).drop(["time"], axis=1)

currentDataFrame = getValuesV2(totalTags, currentStartTime, currenEndTime).drop(
    ["time"], axis=1
)

target = "outputTags"

transformer = pickle.load(open(transFileName, "rb"))
inputDftransRef = transformer.transform(refDataFrame)
annModel = keras.models.load_model(h5FileName)
yPredRef = annModel.predict(inputDftransRef)
yPredDfRef = pd.DataFrame(yPredRef, columns=["prediction"]).reset_index(drop=True)

finalRef = pd.concat([refDataFrame, yPredDfRef], axis=1)
finalRef = finalRef.rename(columns={outputTags[0]: "target"})


inputDftransCur = transformer.transform(currentDataFrame)
annModel = keras.models.load_model(h5FileName)
yPredCur = annModel.predict(inputDftransCur)

yPredCur = pd.DataFrame(yPredCur, columns=["prediction"]).reset_index(drop=True)

finalCur = pd.concat([currentDataFrame, yPredCur], axis=1)

finalCur = finalCur.rename(columns={outputTags[0]: "target"})


reportData = Report(metrics=[DataDriftPreset(), DataDriftTable(), DataQualityPreset()])

# reportModel = Report(metrics=[RegressionPerformanceMetrics(), RegressionPredictedVsActualPlot(), RegressionErrorDistribution()])
reportModel = Report(metrics=[RegressionPreset()])

savingname = modelId + "_Version" + str(currentVersion)

reportData.run(reference_data=finalRef, current_data=finalCur)
reportData.save_html(savingname + "data.html")

reportModel.run(reference_data=finalRef, current_data=finalCur)
reportModel.save_html(savingname + "model.html")


uploadTrainingResults(
    "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "data.html"
)
uploadTrainingResults(
    "C:\\karyalay\\ModelDiagnostics\\main\\", savingname + "model.html"
)
