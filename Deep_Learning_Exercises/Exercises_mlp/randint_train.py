
# general imports
import numpy as np
import matplotlib.pyplot as plt
import pdb

#keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

input_dim = 784                                                     # set the dimensionality of data such as mnist 28 x 28 images

model = Sequential()                                                # Initialize Keras model as Sequential since this is a simple model (in keras there are 2 types of model: Sequential and Model for more complicated models    
model.add(Dense(32, activation='relu', input_dim=input_dim))
model.add(Dense(10, activation='softmax'))                          # softmax is actually a normalization e^y/sum(e^y) which produces probabilities
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
              
# Generate dummy data
data = np.random.randint(255, size=(60000, input_dim))
labels = np.random.randint(10, size=(60000, 1))

# Normalization
data = data / 255.0                                                  # for faster convergece of sgd

# Convert labels to categorical one-hot encoding
labels = to_categorical(labels, num_classes=10)                      # translate to a matrix 60000 x 10, with 5th column = 1 if the label is 5, the 8th col = 1 if label = 8, otherwise is 0
                                                                     # this is because: output from neural network is a vector of 10 probabilities whereas label vectors is made of scalars (single values), for example 5,4,2,8
                                                                     # in order to calculate derivatives we have to translate labels vector in one-hot encoding: 8 becomes [0,0,0,0,0,0,0,1,0,0], this is what to_categorical does
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, validation_split = 0.2, epochs=10, batch_size=32)       