import json
import os
import platform
import re
import time as t
from datetime import date, datetime, time, timedelta
from pprint import pprint as pp

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import requests
from fpdf import FPDF
from logzero import logger
from pytz import timezone


version = platform.python_version().split(".")[0]
if version == "3":
    import app_config.app_config as cfg
elif version == "2":
    import app_config as cfg
config = cfg.getconfig()

UNITD = os.environ.get("UNIT_ID") if os.environ.get("UNIT_ID") is not None else None
if UNITD is None:
    UNITD = "62f3a6d6f38f4206da2bf0a5"  # "62f3a6ebf38f4206da2bf0a7" #Malakoff #"5f0ff2f892affe3a28ebb1c2"
    logger.info("no unit id passed, using default")


unitsId = "62f3a6d6f38f4206da2bf0a5"


def getModelFromId(modelId):
    """This function is use to find machine learning moddel."""
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
    """This function is use to get all the input and output tags from the models fetched."""
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
    """This function is used to get version history of a particular model"""
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


def gettingTimeStamp(date):
    """This function is used for setting time stamps from epochs"""
    date = datetime.strptime(date, "%d-%m-%YT%H:%M:%S")
    timestamp = datetime.timestamp(date)
    return timestamp


def getValuesV2(tagList, startTime, endTime):
    """Getting values from given list of tags"""
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


modelId = "63a292cdf2a0f8000741e8aa"


modelDetails = getModelFromId(modelId)


# getting model details:
modelTags = getTagsFromModelDetailes(modelDetails)

inputTags = modelTags["inputTags"]
outputTags = modelTags["outputTags"]


totalTags = inputTags + outputTags

# -----------------------------model tags and eqp status data fetch-------------------------------------------


def getEqpStatusTag(query):
    tagmeta_uri = (
        config["api"]["meta"]
        + '/tagmeta?filter={"where":'
        + json.dumps(query)
        + ' , "fields": ["equipmentId"]}'
    )

    response = requests.get(tagmeta_uri)

    if response.status_code == 200:
        eqpResp = json.loads(response.content)
    else:
        print(response.status_code)
        print(response.content)

    eqpId = eqpResp[0]["equipmentId"]

    eqp_status = "state__" + str(eqpId)

    return eqp_status


eqpStatusTag = getEqpStatusTag({"dataTagId": inputTags[0]})
versionHist, currentVersion = getLatestVersionHistoryFromModelId(modelId)
modelInputTags = totalTags + [eqpStatusTag]


today = datetime.today()
v = datetime.combine(today, time.min)
l = v - timedelta(days=30)
currentStartTime = (
    int(t.mktime(l.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000
)
currenEndTime = int(t.mktime(v.timetuple())) * 1000 - int(5.5 * 60 * 60 * 1000) - 1000


# replace with desired start and endtimes
df_inputs = getValuesV2(modelInputTags, currentStartTime, currenEndTime)
df_inputs["time"] = pd.to_datetime(df_inputs["time"], unit="ms")



df_time = df_inputs.loc[df_inputs[eqpStatusTag] == 1]

print(df_time)
