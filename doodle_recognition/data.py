#import glob
#import urllib.request

import requests
import io
import os

import numpy as np
import pandas as pd

import tensorflow as tf

from google.cloud import storage

from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES, URL_FOLDER


def create_df(CLASSES, max_items_per_class= 40):
    all_files = []
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in CLASSES:
        cls_url = c.replace(' ', '%20')
        path = url+cls_url+'.npy'
        all_files.append(path)

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load a subset of the data to memory
    for idx, f in enumerate(all_files):
        response = requests.get(f)
        data = np.load(io.BytesIO(response.content))
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        X = np.concatenate((X, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(f))
        class_names.append(class_name)

    #X = np.save(f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/X", X)
    #y = np.save(os.path.join(url_folder,'y.npy'), y)
    #class_names = np.save(os.path.join(url_folder,'class_names.npy'), class_names)

    data = None
    labels = None

    return X, y, class_names

class Preproc_df():
    def fit(self, X, y=None):
        print("dans le fit preproc_df:")
        print(X.shape)
        return self

    def transform(self, X, y=None):
        X /= 255
        X = X.reshape(len(X),28, 28,1)
        print('reshape in transform preproc_df')
        print(X.shape)
        return X

class To_Cat():
    # def __init__(self, num_classes=NUM_CLASSES):
    #     self.NUM_CLASSES = num_classes
    #     print(self)

    def fit(self, X, y):
        print("dans le fit TO_CAT:")
        print(y.shape)
        self.NUM_CLASSES = NUM_CLASSES
        print(self.NUM_CLASSES)

        print("reshape in transform TO_CAT _ before to_cat:")
        print(y.shape)
        self.y = to_categorical(y, num_classes=self.NUM_CLASSES)
        print("reshape in transform TO_CAT:")
        print(self.y.shape)

        return self

    def transform(self, X, y=None):
        return self.y

def create_train_test_val(X,y):
    #separate into training and testing
    permutation = np.random.permutation(y.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    #train_ratio = 0.75
    #validation_ratio = 0.15
    #test_ratio = 0.10

    X_train, X_test_1, y_train,  y_test_1 = train_test_split(X, y, test_size=0.25)
    X_test, X_val, y_test, y_val = train_test_split(X_test_1, y_test_1, test_size=0.30)


    return X_train, y_train, X_test, y_test, X_val, y_val

if __name__ == '__main__':

    X, y, class_names = create_df(CLASSES)
    # X = preproc_df(X)
    # y = to_cat(y, num_classes)
    # X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val(X,y)

