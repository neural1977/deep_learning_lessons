
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

input_dim = 784

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=input_dim))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
              
# Generate dummy data
data = np.random.randint(255, size=(60000, input_dim))
labels = np.random.randint(10, size=(60000, 1))

# Normalization
data = data / 255.0

# Convert labels to categorical one-hot encoding
labels = to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, validation_split = 0.2, epochs=10, batch_size=32)