#Aurelio Arango, Tutorial
#Tutorial from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
import numpy as np
np.random.seed(123)

# Keras model module
from keras.models import Sequential

# Keras core layers
from keras.layers import Dense, Dropout, Flatten, Activation

# Keras CNN layers
from keras.layers import Convolution2D, MaxPooling2D, Conv2D

# Keras utilities
from keras.utils import np_utils

# Load MNISt dataset
from keras.datasets import mnist

from keras import backend as k

# load and shuffle data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print x_train.shape
# from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

imwid = 28
imlen = 28

if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, imwid, imlen)
    x_test = x_test.reshape(x_test.shape[0], 1, imwid, imlen)
    input_shape = (1, imwid, imlen)
else:
    x_train = x_train.reshape(x_train.shape[0], imwid, imlen, 1)
    x_test = x_test.reshape(x_test.shape[0], imwid, imlen, 1)
    input_shape = ( imwid, imlen, 1)

# nomalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Create Sequential model
model = Sequential()


# CNN input layer
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape= input_shape))

# Adding more layers
model.add(Conv2D(64, 3, 3, activation='relu'))
# Slide a 2x2 filter and takes the max of the previous layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Dropout regularizes model to prevent overfitting
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Fit Keras model
model.fit(x_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)
# evaluate model
score = model.evaluate(x_test, Y_test, verbose=0)