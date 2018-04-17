# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:40:01 2018

@author: Nitisha Agarwal
"""
# Plot ad hoc mnist instances
from keras.datasets import mnist
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], 1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1,28,28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


m = larger_model()
m.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 200, batch_size = 200, verbose = 2)
score = m.evaluate(X_test, y_test, verbose = 0)

m.save("C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\bettercnn.h5")
#---- run from here ---- ##############################
from keras.models import load_model
m = load_model("C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\bettercnn.h5")

import cv2


gray = cv2.imread("C:\\Users\\Nitisha Agarwal\\Desktop\\minor2\\crop5.png")
gray = cv2.resize(255-gray, (28, 28))
gray = cv2.cvtColor( gray, cv2.COLOR_RGB2GRAY )
cv2.imwrite( "grey.png", gray )

# convert colored to grayscaled
#gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
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

print("predicted digit: " + str( m.predict_classes(im)[0] ))








# =============================================================================
# 
# 
# 
# # build the model
# m = larger_model()
# 
# 
# # Fit the model
# m.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)
# score=m.evaluate(X_test,y_test,verbose=0)
# m.save("bettercnn.h5")
# 
# 
# m = load_model("bettercnn.h5")
# 
# import cv2
# 
# 
# gray = cv2.imread("4.png")
# gray = cv2.resize(255-gray, (28, 28))
# # convert colored to grayscaled
# gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# if gray[0][0] == 255:
#     gray = 255 - gray
# 
# # save the processed / resized image
# cv2.imwrite("#extra.png", gray)
# 
# 
# import imageio
# im = imageio.imread("#extra.png")
# print (im.size)
#     # reshape and normalise
# im = im.reshape(1, 1, 28, 28).astype('float32')
# im = im / 255.0 
# 
# print("predicted digit: " + str( m.predict_classes(im)[0] ))
# =============================================================================
