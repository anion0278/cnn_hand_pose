#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split

class CnnModel:
    def __init__(self, conv_filters, learning_rate, image_sqr_size, existing_model_name=None):
        if (existing_model_name == None):
            self.model = self.__create_model(conv_filters, learning_rate, image_sqr_size)
        else:
            self.model = tf.keras.models.load_model(existing_model_name)
            self.setup_graph()
        #self.session = None
        #self.graph = None

    def train(self, X_dataset, y_dataset, nb_epoch, batch_size, logs_path, test_data_ratio):

        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=test_data_ratio, random_state=42)

        print("Train data length: %s" % len(X_train))
        print("Test data length: %s" % len(X_test))

        train_datagen = ImageDataGenerator(rescale=1. / 255) # change 0..255 to 0..1

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        tensorboard = TensorBoard(log_dir=logs_path)

        #validation_split=0.20 - instead of splitting, but the data has to be
        #shuffled beforehand!
        self.model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            #steps_per_epoch=40, # if not defined -> will train
                            #use
                            #exactly x_train.size/batch_size
                            epochs=nb_epoch,
                            validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
                            #validation_steps=20, # if not defined -> fill run
                            #all valid data once
                            verbose=1,
                            callbacks=[tensorboard])
        self.setup_graph()

    def predict_image(self, X_img, y_expected):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        testData = test_datagen.flow(X_img, y_expected, batch_size=1)
        with session.as_default():
            with graph.as_default():
                prediction = self.model.predict(testData)
        return prediction

    def setup_graph(self):
        self.model._make_predict_function()
        global session
        session = tf.keras.backend.get_session()
        global graph
        graph = tf.get_default_graph()    

    def save(self, model_name):
        self.model.save(model_name)
        print("Model saved: %s" % model_name) 

    def __create_model(self, conv_filters, learning_rate, image_sqr_size):
        conv_kernel = (3, 3)
        pooling_kernel = (2, 2)
        relu_activation = 'relu'
        input_shape = (image_sqr_size, image_sqr_size, 1)

        model = Sequential()

        model.add(Conv2D(conv_filters, kernel_size=conv_kernel, activation=relu_activation, input_shape=input_shape))

        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(BatchNormalization()) # axis should be set to Channels dimension (width, height, channels),
                                        # default is -1 (the last dimension)

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

        # try model.add(LeakyReLU(alpha=0.05))

        model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, activation=relu_activation)) 
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(conv_filters * 4, activation=relu_activation))
        model.add(Dropout(0.5))

        model.add(Dense(5, activation="linear")) 

        opt = Adam(learning_rate=learning_rate)
        #opt = Adadelta(learning_rate=1.0, rho=0.95)

        model.compile(loss="mean_squared_error", optimizer=opt)
        return model
