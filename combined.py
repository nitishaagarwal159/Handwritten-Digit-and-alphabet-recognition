# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:39:53 2018

@author: Nitisha Agarwal
"""

import os
import struct
import numpy as np

def read(dataset = "training", path = ""):
    if dataset is "training":
        fname_img = os.path.join(path, 'C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\train-images')
        fname_lbl = os.path.join(path, 'C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\train-labels')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\test-images')
        fname_lbl = os.path.join(path, 'C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\test-labels')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (img[idx], lbl[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

training_data = list(read(dataset = 'training', path = ''))
print(len(training_data))
X_train, y_train = [], []
for i in range(len(training_data)):
    X, y = training_data[i]
    X_train.append(X)
    y_train.append(y)

test_data = list(read(dataset = 'testing', path = ''))
print(len(test_data))
X_test, y_test = [], []
for i in range(len(test_data)):
    X, y = test_data[i]
    X_test.append(X)
    y_test.append(y)

show(X_test[3201])
print(y_test[3201])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

np.random.seed(7)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
    m = Sequential()
    m.add(Conv2D(30, (5, 5), input_shape = (1, 28, 28), activation = 'relu'))
    m.add(MaxPooling2D(pool_size = (2, 2)))
    m.add(Conv2D(15, (3, 3), activation = 'relu'))
    m.add(MaxPooling2D(pool_size = (2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation = 'relu'))
    m.add(Dense(50, activation = 'relu'))
    m.add(Dense(num_classes, activation = 'softmax'))
    m.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return m

m = larger_model()
m.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 2, batch_size = 200)
score = m.evaluate(X_test, y_test, verbose = 0)
m.save("beercnn.h5")

print('Large CNN error: %.2f%%' % (100 - score[1] * 100))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

# ----- run -----
from keras.models import load_model
m = load_model("beercnn.h5")    

import cv2
from scipy.ndimage import rotate

gray = cv2.imread("a.jpg")
gray = rotate(gray, 90)
gray = cv2.resize(255-gray, (28, 28))
# convert colored to grayscaled
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
if gray[0][0] == 255:
    gray = 255 - gray

# save the processed / resized image
cv2.imwrite("#extra.png", gray)


import imageio
im = imageio.imread("#extra.png")
print (im.size)
    # reshape and normalise
im = im.reshape(1, 1, 28, 28).astype('float32')
im = im / 255.0 

print("predicted alphabet: " + chr( m.predict_classes(im)[0] + 96 ))