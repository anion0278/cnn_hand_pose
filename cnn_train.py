#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print("Python %s" % sys.version_info[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from os import listdir
from time import time
import numpy as np
import os
import tensorflow as tf
import cv2

image_size = 128 


def parse_expected_value(img_name, name_base):
    result = []
    for fingerState in img_name.replace(name_base, '')[0:5]:
        result.append(int(fingerState))
    return result


def load_train_data(name_pattern):
    X_train = []
    y_train = []
    img_names_by_X = dict()
    for trainImgPath in list(filter(lambda x: x.startswith(name_pattern) and len(x) > 12, listdir())):
        img = cv2.resize(cv2.imread(trainImgPath, cv2.IMREAD_GRAYSCALE),
                         (image_size, image_size)).astype(np.float32)
        img = img[..., np.newaxis]
        img_names_by_X[trainImgPath] = img
        X_train.append(img)
        expected_value = parse_expected_value(trainImgPath, name_pattern)
        y_train.append(expected_value)
        print(trainImgPath + ": " + str(expected_value))
    return X_train, y_train, img_names_by_X


def get_train_data_by_name_pattern(name_pattern):
    train_X, train_y, train_names_by_X = load_train_data(name_pattern)
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    m = train_X.mean()
    s = train_X.std()

    print('Samples mean, sd:', m, s)
    train_X -= m
    train_X /= s
    print('Samples shape:', train_X.shape)
    print(train_X.shape[0], 'normalized samples')
    return train_X, train_y, train_names_by_X


def create_model():
    nb_filters = 16
    nb_kernel = (3, 3)

    model = Sequential()

    model.add(Conv2D(nb_filters, kernel_size=nb_kernel,
                     activation='relu', input_shape=(image_size, image_size, 1)))
    # model.add(BatchNormalization())

    model.add(Conv2D(nb_filters, kernel_size=nb_kernel, activation='relu'))
    # model.add(BatchNormalization())

    model.add(Conv2D(nb_filters * 2, kernel_size=nb_kernel, activation='relu'))
    # model.add(BatchNormalization())

    model.add(Conv2D(nb_filters * 2, kernel_size=nb_kernel, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(nb_filters * 2, kernel_size=nb_kernel, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(nb_filters * 4, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def train_model(nb_epoch):
    batch_size = 32

    print("Train data:")
    X_all, y_all, train_names_by_X = get_train_data_by_name_pattern('state')

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.05,
        zoom_range=0.05,
        # width_shift_range=0.1,
        height_shift_range=0.05)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

    model = create_model()

    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=500,
                        epochs=nb_epoch,
                        validation_data=test_datagen.flow(
                            X_test, y_test, batch_size=batch_size),
                        validation_steps=400,
                        verbose=1, callbacks=[tensorboard])

    model.save('model.h5')
    testData = test_datagen.flow(X_test[0:1], y_test[0:1], batch_size=1)

    return model


def predict(img_name):
    print(img_name)
    tf.keras.models.load_model('model_v1.h5')
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # testData = test_datagen.flow(X_test[0:1], y_test[0:1], batch_size=1)
    pass


if __name__ == "__main__":
    print('main')
    if (sys.argv[1] == '-h'):
        print("    -h - this help")
        print("    -train - train model")
        print("    -predict *fileName* - predict result for file with name")
    if (sys.argv[1] == 'train'):
        train_model(nb_epoch=20)
    if (sys.argv[1] == 'predict' and not(sys.argv[2].isspace())):
        predict(sys.argv[2])
