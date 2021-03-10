
#import glob
#import urllib.request

import requests
import io
import os

import numpy as np
import pandas as pd
from joblib import dump

from google.cloud import storage

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES
from doodle_recognition.data import create_df, Preproc_df, To_Cat, create_train_test_val


def init_model():
    model = Sequential()
    model.add(layers.Conv2D(16, (4,4), strides=(2,2), input_shape=(28, 28, 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Conv2D(32, (3,3), strides=(2,2), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
    # model.fit(X_train, y_train,
                    #batch_size = 32,
                    #epochs=1,
                    #validation_data = (X_test, y_test),
                    #callbacks=[es]
                    # )

    #dump(model, 'model.joblib')
    return model

    # print(colored("model.joblib saved locally", "green"))

    # client = storage.Client()

    # bucket = client.bucket(BUCKET_NAME)

    # blob = bucket.blob(STORAGE_LOCATION)

    # blob.upload_from_filename('model.joblib')

    # print("uploaded model.joblib to gcp cloud storage under \n => {}".format(STORAGE_LOCATION))


def evaluate(model, X_val, y_val):
    """evaluates the pipeline on df_test and return the RMSE"""
    y_pred = model.predict(X_val)
    rmse = np.sqrt(((y_pred - y_val) ** 2).mean())
    return round(rmse, 2)

def save_model(model):
    """Save the model into a .joblib format"""
    dump(model,'model.joblib')
    print(colored("model.joblib saved locally", "green"))

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')

    print("uploaded model.joblib to gcp cloud storage under \n => {}".format(STORAGE_LOCATION))


if __name__ == "__main__":
    # Get and clean data
    init_model()