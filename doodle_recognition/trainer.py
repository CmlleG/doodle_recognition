import requests
import io
import os

import numpy as np
import pandas as pd
from joblib import dump
import pickle as pkl

from sklearn.pipeline import Pipeline
from google.cloud import storage

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.compose import ColumnTransformer

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES, STORAGE_LOCATION, URL_FOLDER
from doodle_recognition.data import create_df, Preproc_df, To_Cat, create_train_test_val
from doodle_recognition.model import init_model, evaluate, save_model, upload_model_to_gcp

class Trainer(object):
    def __init__(self,X,y):

        self.pipeline = None
        self.X = X
        self.y = y
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
        print(self.X.shape)
        print(self.y.shape)

        # prepro_y = Pipeline([
        #     ('prepro_y', To_Cat())])

        # """   preproc_pipe = ColumnTransformer([
        #     ('preprocess_X', prepro_X, range(0,784))
        #     ]) """

        self.pipeline = Pipeline([
            ('prepro', prepro_X),
            ('model', init_model())
        ])

    def run(self):
        self.set_pipeline()
        print("dans le run")
        print(self.X.shape)
        print(self.y.shape)
        self.pipeline.fit(self.X, self.y)


    # def run(self, X_train, y_train, X_test, y_test):
    #     self.pipeline.fit(X_train, y_train,
    #                       batch_size = 32,
    #                       epochs=100,
    #                       validation_data = (X_test, y_test),
    #                       callbacks=[es]
    #                       )

    # def evaluate(self, X_val, y_val):
    #     """evaluates the pipeline on df_test and return the RMSE"""
    #     y_pred = self.pipeline.predict(X_val)
    #     rmse = np.sqrt(((y_pred - y_val) ** 2).mean())
    #     return round(rmse, 2)

    def save_model(self):
        """Save the model into a .joblib format"""
        print("model.joblib saved locally")
        dump(self.pipeline, 'model.joblib')

    def upload_model_to_gcp(self):

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')

        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(STORAGE_LOCATION))

if __name__ == "__main__":

    X, y, class_names = create_df(CLASSES)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    # print("dans le trainer")
    # print(X.shape)
    # print(y.shape)
    # X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val(X,y)
    # print('train')
    # print(X_train.shape)
    # print(y_train.shape)
    # print('test')
    # print(X_test.shape)
    # print(y_test.shape)
    # print('val')
    # print(X_val.shape)
    # print(y_val.shape)

    trainer = Trainer(X=X ,y=y)
    # print("dans le trainer, apres trainer")
    # print(X.shape)
    # print(y.shape)
    trainer.run()
    trainer.save_model()









    # Get and clean data
    #es = EarlyStopping(patience=10, verbose=0)



    #X = preproc_df(X)
    #y = to_cat(y, NUM_CLASSES)


    #model = init_model(X_train=X_train, y_train=y_train)
    #print(model)
    #model = model.fit(X_train, y_train,
                      #batch_size = 32,
                      #epochs=1,
                      #validation_data = (X_test, y_test),
                      #callbacks=[es]
                      #)



    #model.save_model()
    #upload_model_to_gcp(model)
    #trainer = Trainer(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
    #trainer.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


    # rmse = trainer.evaluate(X_val=X_val, y_val=y_val)
    # print(f"rmse: {rmse}")

    # trainer.save_model()
    # trainer.upload_model_to_gcp()
