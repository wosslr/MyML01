# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np

from MyML_test11_get_ckpt_files import get_ckpt_file_seir, get_birds_files
import argparse

# parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
# parser.add_argument('image', type=str, help='The image image file to check')
# args = parser.parse_args()


# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')
# model.load("bird-classifier.tfl.ckpt-50912")

ckptseries = get_ckpt_file_seir()
good_seris = []

for seri in ckptseries:
    ckpt_file = "bird-classifier.tfl.ckpt-" + seri
    model.load(ckpt_file)
    birds_images = get_birds_files()
    imgs = []
    for birds_image in birds_images:
        img = scipy.ndimage.imread(birds_image, mode="RGB")
        img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
        imgs.append(img)
    prediction = model.predict(imgs)
    good_count = 0
    for pre in prediction:
        if np.argmax(pre) == 1:
            good_count += 1
    if good_count / len(prediction) > 0.7:
        good_seris.append({
            'seri': seri,
            'good_count': good_count
        })

print(good_seris)