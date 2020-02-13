from keras.layers import Dense, Input, Reshape
from keras.models import Sequential, Model
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random 
#random.seed(42)
import pdb

inp = Input(shape=((28,28,)))
resh = Reshape((784,))(inp)
layer1 = Dense(32, activation='relu')(resh)
layer2 = Dense(10, activation='softmax')(layer1)
model = Model(input = inp, output = layer2)

#model = Sequential()
#model.add(Dense(32, activation='relu', input_dim=(100)))
#model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# Generate dummy data
np.random.seed(42)
#data = np.random.random((1000, 100))
#labels = np.random.randint(2, size=(1000, 1))
#pdb.set_trace()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#pdb.set_trace()
# Train the model, iterating on the data in batches of 32 samples
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split = 0.2, verbose = 2)

score = model.evaluate(x_test, y_test, batch_size=5000)
pred = model.predict(x_test, batch_size=5000)