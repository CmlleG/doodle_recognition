
import requests
import io
import os

import numpy as np
import pandas as pd
from joblib import dump

from google.cloud import storage

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import models

from tensorflow.keras import layers
#from tensorflow import keras
#import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES, BUCKET_FOLDER_R, BUCKET_FOLDER_M, VERSION,STORAGE_LOCATION, DATA_FOLDER
from doodle_recognition.data import create_df, create_train_test_val


def init_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (4,4), strides=(2,2), input_shape=(28, 28, 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Conv2D(32, (3,3), strides=(2,2), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
    return model

def initialize_model():
    model = models.Sequential()
    model.add(layers.Convolution2D(16, (3, 3),
                            padding='same',
                            input_shape=(28,28,1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size =(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    # Train model
    model.summary()
    #adam = tf.compat.v1.train.AdamOptimizer()
    f1 = tfa.metrics.F1Score(num_classes=NUM_CLASSES,average = 'macro')
    kappa = tfa.metrics.CohenKappa(num_classes=NUM_CLASSES)
    top = metrics.TopKCategoricalAccuracy(k=1)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy', kappa,f1,top])
    return model

def save_model_to_gcp(joblib_name):

    storage_location = f"models/{joblib_name}"
    local_model_filename = joblib_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{joblib_name}")
    blob.upload_from_filename(local_model_filename)

def save_image_to_gcp(fig_name):

    storage_location = f"result/{fig_name}"
    local_model_filename = DATA_FOLDER_R + '/' + fig_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)

def save_classification_report_gcp(class_name):

    storage_location = f"classification_reports/{class_name}"
    local_model_filename = DATA_FOLDER_R + '/' + fig_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)


if __name__ == "__main__":
    # Get and clean data
    init_model()