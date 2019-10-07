# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from os import listdir
from time import time
import numpy as np
import os
import tensorflow as tf

import sys
import math
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

image_size = 128
image_base = 'state'

def parse_expected_value(img_name, name_base = image_base):
    result = []
    for fingerState in img_name.replace(name_base, '')[0:5]:
        result.append(int(fingerState))
    return result


def load_single_img(img_name):
    X_img = cv2.resize(cv2.imread(img_name, cv2.IMREAD_GRAYSCALE),
                     (image_size, image_size)).astype(np.float32)
    X_img = X_img[..., np.newaxis]
    y_expected = parse_expected_value(img_name)
    print("Loaded: "+ img_name + " -> " + str(y_expected))
    return X_img, y_expected


def get_train_data():
    X_train = []
    y_train = []
    img_names_by_X = dict()
    for train_img_path in list(filter(lambda x: x.startswith(image_base) and len(x) > 12, listdir())):
        X_img, y_expected = load_single_img(train_img_path)
        X_train.append(X_img)
        y_train.append(y_train)
        # img = cv2.resize(cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE),
        #                  (image_size, image_size)).astype(np.float32)
        # img = img[..., np.newaxis]
        # img_names_by_X[train_img_path] = img
        # X_train.append(img)
        # expected_value = parse_expected_value(train_img_path, name_pattern)
        # y_train.append(expected_value)
        # print(train_img_path + ": " + str(expected_value))
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), img_names_by_X


# def get_train_data_by_name_pattern(name_pattern):
#     train_X, train_y, train_names_by_X = get_train_data(name_pattern)
#     train_X = np.array(train_X, dtype=np.float32)
#     train_y = np.array(train_y, dtype=np.float32)

#     print('Samples shape:', train_X.shape)
#     print(train_X.shape[0], 'normalized samples')
#     return train_X, train_y, train_names_by_X


X_all, y_all, train_names_by_X = get_train_data()
print('Samples shape:', X_all.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42)

model = tf.keras.models.load_model('model_v1.h5')

# test change
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255)

# X_sample = []
# y_sample = []
# sample_name = 'state00001_img13_date06-10-2019 20#50#27.jpeg'
# img = cv2.resize(cv2.imread(sample_name, cv2.IMREAD_GRAYSCALE),
#                  (image_size, image_size)).astype(np.float32)
# img = img[..., np.newaxis]
# X_sample.append(img)
# X_sample.append(img)
# expected_value = parse_expected_value(sample_name, 'state')
# y_sample.append(expected_value)

X_data, y_data = test_datagen.flow(X_test, y_test, batch_size=1).next()


predict = np.round(model.predict(X_data), 0)
print('expected: %s - predicted: %s' % (y_data, predict))
