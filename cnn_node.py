#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
ros_packages = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if (ros_packages in sys.path):
    sys.path.remove(ros_packages)
print("Python %s" % sys.version_info[0])

import socket
import threading
import socketserver
import time
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from os import listdir
from time import time

image_size = 128
current_script_path = os.path.dirname(os.path.realpath(__file__))
current_model = os.path.join(current_script_path,"model_v3.h5")
image_base = "state"
dataset_dir = "dataset"



def parse_expected_value(img_name, name_base=image_base):
    result = []
    state_start = img_name.find(image_base)
    state = img_name[state_start:].replace(image_base, "")
    for fingerState in state[0:5]:
        result.append(int(fingerState))
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
    for train_img_path in list(filter(lambda x: x.startswith(image_base) and len(x) > 12, listdir(dataset_dir))):
        X_img, y_expected = load_single_img(os.path.join(dataset_dir, train_img_path))
        X_train.append(X_img)
        y_train.append(y_expected)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


def create_model():
    conv_filters = 16
    conv_kernel = (3, 3)
    pooling_kernel = (2, 2)
    relu_activation = 'relu'

    model = Sequential()

    model.add(Conv2D(conv_filters, kernel_size=conv_kernel,
                     activation=relu_activation, input_shape=(image_size, image_size, 1)))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation))
    model.add(MaxPooling2D(pool_size=pooling_kernel))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(conv_filters * 4, activation=relu_activation))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation="sigmoid"))

    opt = Adam(learning_rate=0.0001)

    model.compile(loss="mean_squared_error", optimizer=opt)
    return model


def train_model(nb_epoch):
    batch_size = 128

    X_all, y_all = get_train_data()

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

    print("Train data length: %s" % len(X_train))
    print("Test data length: %s" % len(X_test))

    train_datagen = ImageDataGenerator(rescale=1. / 255,
        shear_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.05)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    current_log_dir = os.path.join(current_script_path, 'logs', format(time()))
    tensorboard = TensorBoard(log_dir=current_log_dir)

    model = create_model()

    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=500,
                        epochs=nb_epoch,
                        validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
                        validation_steps=400,
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
    predicted = np.round(model.predict(testData), 0)
    print("expected: %s - predicted: %s" % (expected_result, predicted))
    print("Keras Result: %s" % predicted)
    return predicted


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
    result = np.round(prediction, 0)
    print("expected: %s - predicted: %s" % (expected_result, result))
    print("Keras Result: %s" % result)
    return result

def service_mode():
    model_for_server = tf.keras.models.load_model(current_model)    
    ip, port = "localhost", 11001
    server = ThreadedTCPServer((ip, port), ThreadedTCPRequestHandler, model_for_server)
    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)

    while (True):
        pass

server_command = "PredictImage:"

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            data = str(self.request.recv(1024), 'ascii')
            cur_thread = threading.current_thread()
            print("Server recieved %s" % data)
    
            responce = "unknown command"
            if (data.find(server_command) != -1):
                responce = predict_using_model(self.server.model, data.replace(server_command, ""))

            response = bytes("{}: {}".format(cur_thread.name, responce), 'ascii')
            self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, model, bind_and_activate=True):
        self.model = model
        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate=True)


if __name__ == "__main__":
   
    print(sys.argv)

    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == "-h"):
        print("    -h - this help")
        print("    -train - train model")
        print("    -predict *fileName* - predict result for file with name")

    if (sys.argv[1] == "train"):
        print("Training...")
        train_model(nb_epoch=20)

    if (sys.argv[1] == "predict" and not(sys.argv[2].isspace())):
        print("Predicting %s" % sys.argv[2])
        predict_single_img(sys.argv[2])
   
    if (sys.argv[1] == "predict_service"):
        model_for_server = tf.keras.models.load_model(current_model)
        model_for_server._make_predict_function()

        global session
        session = tf.keras.backend.get_session()

        global graph
        graph = tf.get_default_graph()    
        
        service_mode()
   
