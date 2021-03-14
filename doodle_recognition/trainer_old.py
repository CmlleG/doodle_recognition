import requests
import io
import os

import numpy as np
import pandas as pd
from joblib import dump
import pickle as pkl

# import types
# import tempfile
# import keras.models

from sklearn.pipeline import Pipeline
from google.cloud import storage

from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import models

from tensorflow.keras import layers


from tensorflow.keras.callbacks import EarlyStopping
from sklearn.compose import ColumnTransformer

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES, STORAGE_LOCATION, URL_FOLDER, NUM_CLASSES_TEST, CLASSES_TEST
from doodle_recognition.data import create_df, Preproc_df, create_train_test_val, one_df, save_target
from doodle_recognition.model import init_model, initialize_model

class Trainer(object):
    def __init__(self, X_train, y_train, X_test, y_test, es):

        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.es = es
        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

    # def set_experiment_name(self, experiment_name):
    #     '''defines the experiment name for MLFlow'''
    #     self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        prepro_X = Pipeline([
            ('prepro_X', Preproc_df())])

        print("dans set_pipeline")
        print(self.X_train.shape)
        print(self.y_train.shape)

        self.pipeline = Pipeline([
            ('prepro', prepro_X),
            ('model', initialize_model())
        ])

    def run(self):
        self.set_pipeline()
        print("dans le run")
        print(self.X_train.shape)
        print(self.y_train.shape)

        fit_params = {
            'model__batch_size' : 32,
            'model__epochs' : 500,
            #'model__validation_data' : (self.X_val, self.y_val),
            'model__validation_split' : 0.10,
            'model__callbacks' : [self.es]
        }

        self.pipeline.fit(self.X_train, self.y_train, **fit_params)

    # def evaluate(self, X_val, y_val):
    #     """evaluates the pipeline on df_test and return the RMSE"""
    #     y_pred = self.pipeline.predict(X_val)
    #     rmse = np.sqrt(((y_pred - y_val) ** 2).mean())
    #     return round(rmse, 2)

    def save_model(self):
        """Save the model into a .joblib format"""
        print("model.joblib saved locally")
        #self.save('keras_model.h5')
        dump(self.pipeline, 'model.joblib')

    def upload_model_to_gcp(self):

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')

        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(STORAGE_LOCATION))

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


if __name__ == "__main__":

    # X, y, class_names = one_df(CLASSES_TEST)

    # make_keras_picklable()
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob(STORAGE_LOCATION)
    # es = EarlyStopping(patience=10, verbose=1)

    X, y, class_names = create_df(CLASSES)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    X /= 255
    X = X.reshape(len(X),28, 28,1)
    X_train, y_train, X_test, y_test = create_train_test_val(X,y)
    save_target(y_train, y_test, X_test, class_names)

    # model = initialize_model()
    # model.fit(X_train, y_train,
    #           batch_size = 32,
    #           epochs = 500,
    #           validation_data = (X_test, y_test),
    #           callbacks = [es])
    # model.save("models.joblib",save_format='h5')
    # blob.upload_from_filename('model.joblib')

    # print("dans le trainer")
    # print(X.shape)
    # print(y.shape)
    # es = EarlyStopping(patience=10, verbose=1)
    #X_train, y_train, X_test, y_test = create_train_test_val(X,y)
    # print('train')
    # print(X_train.shape)
    # print(y_train.shape)
    # print('test')
    # print(X_test.shape)
    # print(y_test.shape)

    # trainer = Trainer(X_train=X_train ,y_train=y_train,
    #                   X_test=X_test, y_test=y_test,
    #                   es=es)
    # print("dans le trainer, apres trainer")
    # print(X.shape)
    # print(y.shape)
    # trainer.run()
    # trainer.save_model()
    # trainer.upload_model_to_gcp()