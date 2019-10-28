#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
ros_packages = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if (ros_packages in sys.path):
    sys.path.remove(ros_packages)
print("Python %s" % sys.version_info[0])

import socket
import data_generator
import tensorflow as tf
import threading
import socketserver
import time
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from os import listdir
from time import time
from skimage import exposure
import random

image_size = 128
current_script_path = os.path.dirname(os.path.realpath(__file__))
image_base = "state"
dataset_dir = "data_gen_test"



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
    shear_range=20,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    preprocessing_function = seq.augment_image
    )


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

    train_datagen = ImageDataGenerator(rescale=1. / 255,
        shear_range=0.05,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.05)

def get_train_data():
    X_train = []
    y_train = []
    for train_img_path in list(filter(lambda x: x.startswith(image_base) and len(x) > 12, listdir(dataset_dir))):
        X_img, y_expected = load_single_img(os.path.join(dataset_dir, train_img_path))
        X_train.append(X_img)
        y_train.append(y_expected)
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


import matplotlib.pyplot as plt
img = cv2.resize(cv2.imread('dataset\state00111_img29_date16-10-2019_11#02#04.jpeg', cv2.IMREAD_GRAYSCALE), (image_size, image_size)).astype(np.float32);


img = img[np.newaxis, ..., np.newaxis]
#datagenerator = ImageDataGenerator(horizontal_flip=True) # flips L to R

def AHE(img):
    img_adapteq = exposure.equalize_hist(img)
    #img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

def AHE2(img):
    p2, p98 = np.percentile(img, (25, random.randrange(25, 75)))
    img_adapteq = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_adapteq

def AHE3(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

def contrast_change(img):
    img_new = tf.image.random_contrast(img, 0.7, 1.3).eval()
    return img_new

# zca_whitening=True # does nothing

from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    sometimes(iaa.Add(45)),
    sometimes(iaa.AdditiveGaussianNoise(0.03)),
    sometimes(iaa.Multipy(1.2)),
    sometimes(iaa.AverageBlur(2)),
    sometimes(iaa.SigmoidContrast(7)),
    sometimes(iaa.GammaContrast(1.3)),
    iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
    iaa.PerspectiveTransform(scale=(0.04, 0.08)),
    iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0))])  


datagenerator = ImageDataGenerator(preprocessing_function = seq.augment_image)  #preprocessing_function=AHE2   shear_range=20

for i in range(0,100):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax = ax.ravel()
    ax[0].imshow(np.squeeze(img), cmap='gray')
    ax[0].title.set_text('Original')
    img_grayscale = np.squeeze(next(datagenerator.flow(img))[0])
    ax[1].imshow(img_grayscale, cmap='gray')
    ax[1].title.set_text("Modified")
    fig.suptitle('Featurewise normalization', fontsize=16)
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

mod_val = -0.25  
datagenerator = ImageDataGenerator(height_shift_range=(-mod_val, -mod_val))

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax = ax.ravel()
ax[0].imshow(np.squeeze(img), cmap='gray')
ax[0].title.set_text('Original')
img_grayscale = np.squeeze(next(datagenerator.flow(img))[0])
ax[1].imshow(img_grayscale, cmap='gray')
ax[1].title.set_text("Modified")
fig.suptitle('Height shift', fontsize=16)
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close(fig)


mod_val = 0.25   
datagenerator = ImageDataGenerator(width_shift_range=(-mod_val, -mod_val))

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax = ax.ravel()
ax[0].imshow(np.squeeze(img), cmap='gray')
ax[0].title.set_text('Original')
img_grayscale = np.squeeze(next(datagenerator.flow(img))[0])
ax[1].imshow(img_grayscale, cmap='gray')
ax[1].title.set_text("Modified")
fig.suptitle('Width shift', fontsize=16)
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close(fig)



mod_val = 1.2
datagenerator = ImageDataGenerator(brightness_range=(mod_val, mod_val),)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax = ax.ravel()
ax[0].imshow(np.squeeze(img), cmap='gray')
ax[0].title.set_text('Original')
img_grayscale = np.squeeze(next(datagenerator.flow(img))[0])
ax[1].imshow(img_grayscale, cmap='gray')
ax[1].title.set_text("Modified")
fig.suptitle('Brightness', fontsize=16)
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close(fig)


mod_val = 0.2
datagenerator = ImageDataGenerator(zoom_range=mod_val)

fig, ax = plt.subplots(1, 5, figsize=(10, 10))
ax = ax.ravel()
ax[0].imshow(np.squeeze(img), cmap='gray')
ax[0].title.set_text('Original')
for i in range (1, 5):
    img_grayscale = np.squeeze(next(datagenerator.flow(img))[0])
    ax[i].imshow(img_grayscale, cmap='gray')
    ax[i].title.set_text("Modified: " + str(i))
fig.suptitle('Zoom', fontsize=16)
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close(fig)
