
import requests
import io
import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf

from google.cloud import storage

from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, CATEGORY, NUM_CLASSES, URL_FOLDER, NUM_CLASSES_TEST, CLASSES_TEST, STORAGE_LOCATION,BUCKET_FOLDER_R, DATA_FOLDER, VERSION


def create_df(CLASSES, max_items_per_class= 30000):

    print("-------Start CREATE_DF----------------------------------------")
    all_files = []
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    for c in CLASSES:
        cls_url = c.replace(' ', '%20')
        path = url+cls_url+'.npy'
        all_files.append(path)
    all_files = sorted(all_files)

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    print("---------------download files, done --> starting concat------------------")

    #load a subset of the data to memory
    i = 1
    for idx, f in enumerate(all_files):
        print(f"-------------------------------- file {i} -------------------------------------------")
        response = requests.get(f)
        data = np.load(io.BytesIO(response.content))

        data_squeezed = data[0: max_items_per_class, :]

        labels = np.full(data_squeezed.shape[0], idx)
        # print("size of data_3", sys.getsizeof(data))

        X = np.concatenate((X, data_squeezed), axis=0)
        print("X_after concat", sys.getsizeof(X), type(X), X.shape)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(f))
        class_names.append(class_name)
        i +=1

    data = None
    labels = None
    print("-----------------------------------------------CREATE_DF DONE----------------------------------------")

    return X, y, class_names


def save_df_to_gcp(X_test, y_test, y_train, class_names):

    storage_location = DATA_FOLDER
    print("--------------------------------To pd.DF---------------------")

    y_train_df = pd.DataFrame(y_train, columns=["target"])
    #y_train_df.to_csv('y_train.csv')

    y_test_df = pd.DataFrame(y_test, columns=["target"])
    #y_test_df.to_csv('y_test.csv')

    X_test_df = pd.DataFrame(X_test)
    #X_test_df.to_csv('X_test.csv')

    class_names_df = np.array(class_names)
    class_names_df = pd.DataFrame(class_names_df)
    #class_names_df.to_csv('class_names.csv')

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    #blob = bucket.blob(STORAGE_LOCATION)
    bucket.blob(f"data/X_test_{VERSION}_{CATEGORY}.csv").upload_from_string(X_test_df.to_csv(), 'text/csv')
    bucket.blob(f"data/y_train_{VERSION}_{CATEGORY}.csv").upload_from_string(y_train_df.to_csv(), 'text/csv')
    bucket.blob(f"data/y_test_{VERSION}_{CATEGORY}.csv").upload_from_string(y_test_df.to_csv(), 'text/csv')
    bucket.blob(f"data/class_names_{VERSION}_{CATEGORY}.csv").upload_from_string(class_names_df.to_csv(), 'text/csv')

    #blob.upload_from_filename(y_train_df)
    #blob.upload_from_filename(X_test_df)
    #blob.upload_from_filename(y_test_df)
    #blob.upload_from_filename(class_names_df)


def reshape_X(X):
    X /= 255
    X = X.reshape(len(X),28, 28,1)
    print('reshape in transform preproc_df')
    print(X.shape)
    return X

# class Preproc_df():
#     def fit(self, X, y=None):
#         print("dans le fit preproc_df:")
#         print(X.shape)
#         return self

#     def transform(self, X, y=None):
#         X /= 255
#         X = X.reshape(len(X),28, 28,1)
#         print('reshape in transform preproc_df')
#         print(X.shape)
#         return X

# class To_Cat():
#     # def __init__(self, num_classes=NUM_CLASSES):
#     #     self.NUM_CLASSES = num_classes
#     #     print(self)

#     def fit(self, X, y):
#         print("dans le fit TO_CAT:")
#         print(y.shape)
#         self.NUM_CLASSES = NUM_CLASSES
#         print(self.NUM_CLASSES)

#         print("reshape in transform TO_CAT _ before to_cat:")
#         print(y.shape)
#         self.y = to_categorical(y, num_classes=self.NUM_CLASSES)
#         print("reshape in transform TO_CAT:")
#         print(self.y.shape)

#         return self

#     def transform(self, X, y=None):
#         return self.y

def create_train_test_val(X,y):
    #separate into training and testing
    permutation = np.random.permutation(y.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    #train_ratio = 0.75
    #validation_ratio = 0.15
    #test_ratio = 0.10

    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.05)
    #X_test, X_val, y_test, y_val = train_test_split(X_test_1, y_test_1, test_size=0.30)


    return X_train, y_train, X_test, y_test

if __name__ == '__main__':

    X, y, class_names = create_df(CLASSES)
    # X = preproc_df(X)
    # y = to_cat(y, num_classes)
    # X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val(X,y)

