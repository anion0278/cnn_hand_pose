#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
ros_packages = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if (ros_packages in sys.path):
    sys.path.remove(ros_packages)
print("Python %s" % sys.version_info[0])

import time
import tensorflow as tf
import numpy as np
import os
import cv2
import zmq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
from os import listdir
from time import time
from imgaug import augmenters as iaa
import re

image_size = 128
current_script_path = os.path.dirname(os.path.realpath(__file__))
current_model = os.path.join(current_script_path,"model_actual.h5")
image_state_name = "state"
dataset_dir = "dataset"


def parse_expected_value(img_name, name_base=image_state_name):
    result = []
    match = re.search('.+'+image_state_name+'(\d+)-(\d+)-(\d+)-(\d+)-(\d+)', img_name)
    for fingerIndex in range(0,5):
        y_value = int(match.group(fingerIndex + 1)) / 100
        result.append(y_value)
    return result

def load_single_img(img_name, load_from_train_data=True):
    img_path = os.path.join(current_script_path, img_name)
    print("Loading image from %s" % img_path)
    X_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
                       (image_size, image_size)).astype(np.float32)
    X_img = X_img[..., np.newaxis]
    y_expected = parse_expected_value(img_name)
    print("Loaded: " + img_name + " -> " + str(y_expected))
    return X_img, y_expected


def get_train_data():
    X_train = []
    y_train = []
    for train_img_path in list(filter(lambda x: x.startswith(image_state_name) and len(x) > 12, listdir(dataset_dir))):
        X_img, y_expected = load_single_img(os.path.join(dataset_dir, train_img_path))
        X_train.append(X_img)
        y_train.append(y_expected)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


def create_model():
    conv_filters = 32
    conv_kernel = (3, 3)
    pooling_kernel = (2, 2)
    relu_activation = 'relu'

    model = Sequential()

    model.add(Conv2D(conv_filters, kernel_size=conv_kernel,
                     activation=relu_activation, input_shape=(image_size, image_size, 1)))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization()) # axis should be set to Channels dimension (width, height, channels)
                                    # that we dant want to reduce TODO CHECK!!!
                                                                       # -1 is the last axis

    model.add(Conv2D(conv_filters, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation)) # try model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(conv_filters * 4, activation=relu_activation))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation="linear")) # sigmoid, linear

    opt = Adam(learning_rate=0.0001)
    #opt = Adadelta(learning_rate=1.0, rho=0.95)

    model.compile(loss="mean_squared_error", optimizer=opt)
    return model


def train_model(nb_epoch):
    batch_size = 128

    X_all, y_all = get_train_data()

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.001, random_state=42)

    print("Train data length: %s" % len(X_train))
    print("Test data length: %s" % len(X_test))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([sometimes(iaa.Add(35)),
        sometimes(iaa.AdditiveGaussianNoise(0.03)),
        sometimes(iaa.Multiply(0.8)),
        sometimes(iaa.AverageBlur(2)),
        sometimes(iaa.SigmoidContrast(7)),
        sometimes(iaa.GammaContrast(1.3)),
        iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
        iaa.PerspectiveTransform(scale=(0.04, 0.08)),
        iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0))],
        random_order=True)  

    train_datagen = ImageDataGenerator(rescale=1. / 255, # change 0..255 to 0..1
        #shear_range=20,
        #zoom_range=0.2,
        #rotation_range=30,
        #width_shift_range=0.25,
        #height_shift_range=0.25,
        #preprocessing_function = seq.augment_image
        )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    current_log_dir = os.path.join(current_script_path, 'logs', format(time()))
    tensorboard = TensorBoard(log_dir=current_log_dir)

    model = create_model()

    #validation_split=0.20 - instead of splitting, but the data has to be
    #shuffled beforehand!
    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        #steps_per_epoch=40, # if not defined -> will train use exactly x_train.size/batch_size
                        epochs=nb_epoch,
                        validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
                        #validation_steps=20, # if not defined -> fill run all
                        #the validation data once
                        verbose=1,
                        #use_multiprocessing=True, workers=4
                        callbacks=[tensorboard])

    model.save("model.h5")
    testData = test_datagen.flow(X_test[0:1], y_test[0:1], batch_size=1)
    return model


def predict_single_img(img_name):
    print("Predicting: " + img_name)
    model = tf.keras.models.load_model(current_model)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    img, expected_result = load_single_img(img_name)
    X_predict = np.array([img])
    y_predict = np.array([expected_result])
    testData = test_datagen.flow(X_predict, y_predict, batch_size=1)
    prediction_result = model.predict(testData)
    print("expected: %s - predicted: %s" % (expected_result, prediction_result))
    print("Keras Result: %s" % prediction_result)
    return prediction_result


def predict_using_model(model, img_name):
    print("Predicting: " + img_name)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    img, expected_result = load_single_img(img_name)
    X_predict = np.array([img])
    y_predict = np.array([expected_result])
    testData = test_datagen.flow(X_predict, y_predict, batch_size=1)
    with session.as_default():
        with graph.as_default():
            prediction = model.predict(testData)
    #prediction = model.predict(testData)
    result = np.round(prediction, 2)
    print("expected: %s - predicted: %s" % (expected_result, result))
    print("Keras Result: %s" % result)
    return result

def service_mode():
    model_for_server = tf.keras.models.load_model(current_model)    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = 11001
    socket.bind("tcp://*:%s" % port)
    print("Server started on port:", port)
    server_command = "PredictImage:"
    while (True):
        message = socket.recv_string()
        print("Received request: %s" % message)
        responce = "unknown command"
        if (message.find(server_command) != -1):
            responce = predict_using_model(model_for_server, message.replace(server_command, ""))
        socket.send_string(str(responce))


def setup_predictive_service(model_name):
    print("Setting up service mode")
    model_for_server = tf.keras.models.load_model(model_name)
    model_for_server._make_predict_function()
    global session
    session = tf.keras.backend.get_session()
    global graph
    graph = tf.get_default_graph()    

if __name__ == "__main__":
   
    predictive_service = "predictive_service"

    print(sys.argv) 

    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == "-h"):
        print("    -h - this help")
        print("    -train - train model")
        print("    -predict *fileName* - predict result for file with name")
        sys.exit(1)

    if (sys.argv[1] == "train"):
        print("Training...")
        train_model(nb_epoch=50)
        sys.exit(1)

    if (sys.argv[1] == "predict" and not(sys.argv[2].isspace())):
        print("Predicting %s" % sys.argv[2])
        predict_single_img(sys.argv[2])
        sys.exit(1)
   
    if (len(sys.argv) == 3 and sys.argv[1] == predictive_service):
        setup_predictive_service(sys.argv[2])
        service_mode()
        sys.exit(1)

    if (sys.argv[1] == predictive_service):
        setup_predictive_service(current_model)
        service_mode()
        sys.exit(1)
   
    print("Unknown argument")
   
