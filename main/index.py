from deepchecktest import *
import json
import time
import logging as lg
import paho.mqtt.client as paho
import os

# dataDeviation(" ")


def on_message(client, userdata, msg):
    body = json.loads(msg.payload)
    modelId = body["modelId"]
    dataDeviation(modelId)
    print(body)


def on_connect(client, userdata, flags, rc):
    topic_line = "Optimization/datadeviation/+"
    client.subscribe(topic_line)


def on_log(client, userdata, obj, buff):
    print("log:" + str(buff))


port = os.environ.get("Q_PORT")
if not port:
    port = 1883
else:
    port = int(port)
print("Running port", port)


print(config["BROKER_ADDRESS"])


client = paho.Client()
client.on_log = on_log
client.on_connect = on_connect
client.on_message = on_message


client.connect(config["BROKER_ADDRESS"], port, 60)
client.loop_forever(retry_first_connection=True)
