#!/usr/bin/env python
# -*- coding: utf-8 -*-

import image_data_loader
import tcp_server
import cnn_model
import os
import numpy as np
from time import time

# names

current_model = "model_actual.h5"
command_predict = "PredictImage:"

# model and training
filters_count = 32
learning_rate = 0.0001
batch_size = 128
epochs_count = 5
test_data_ratio = 0.02

# communication
tcp_port = 11001

class PredictorFacade:
    def __init__(self, data_loader, logs_dir,image_size):
        self.loaded_model = None
        self.data_loader = data_loader
        self.logs_dir = logs_dir
        self.image_size = image_size

    def train_model(self, model_name=None): # if model_name is set - continues the learning process
        model = cnn_model.CnnModel(filters_count, learning_rate, self.image_size, model_name)
        X_data, y_data = self.data_loader.get_train_data()
        model.train(X_data, y_data, epochs_count, batch_size, self.logs_dir, test_data_ratio)
        return model

    def predict_single_img(self, img_name):
        self.loaded_model = self.load_model(current_model)
        return self.predict_using_model(self.loaded_model, img_name)

    def predict_using_model(self, model, img_name):
        print("Predicting: " + img_name)
        X_predict, y_predict = self.data_loader.load_single_img(img_name)
        result = np.round(model.predict_image(X_predict, y_predict), 2)
        print("expected: %s - predicted: %s" % (y_predict, result))
        return result

    def service_mode(self):
        server = tcp_server.TcpServer(tcp_port, self.__predictor_callback)
        server.start_server_mode_sync(self.loaded_model)

    def load_model(self, model_name):
        print("Loading model: %s" % model_name)
        self.loaded_model = cnn_model.CnnModel(filters_count, learning_rate, self.image_size, model_name)

    def __predictor_callback(self, model, request):
        print("Received request: %s" % request)
        responce = "Unknown command"
        if (request.find(command_predict) != -1):
            responce = self.predict_using_model(model, request.replace(command_predict, ""))
        return responce