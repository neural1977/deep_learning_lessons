from keras.layers import Dense
from keras.models import Sequential 
import numpy as np
import random 
#random.seed(42)
import pdb

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
np.random.seed(42)
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

#pdb.set_trace()

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32, validation_split = 0.2)