# general imports
import keras
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os

# keras imports
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import array_to_img, img_to_array
from keras.optimizers import SGD

# Program imports
from Models.kconvnet import ConvNet
from Preprocessing.preprocessing import Preprocess

# Set CPU or GPU type
gpu = True
gpu_id = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

default_callbacks = []
input_size = 224

X_train, y_train, X_val, y_val, X_test, y_test = Preprocess.load_belli_brutti('./Input/Utente1',0.1,0.1,True,6,input_size,True)

# Show the first image from the training set
plt.imshow(array_to_img(X_train[0]))
plt.savefig("first_belli_brutti_train_image.jpg")
print("First Belli and Brutti train image", y_train[0])
plt.show(block = False)
plt.pause(3)
plt.close()

# Show the first image from the test set
plt.imshow(array_to_img(X_test[0]))
plt.savefig("first_belli_brutti_test_image.jpg")
print("First Belli and Brutti test image", y_test[0])
plt.show(block = False)
plt.pause(3)
plt.close()

# Normalization (testare come senza normalizzazione converge molto tardi a 95% rispetto alla normalizzazione)
X_train = X_train / 255.0
X_test = X_test / 255.0

model = ConvNet.build(input_size,input_size,3,1,True,None)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['accuracy'])
              
model.summary()

# Convert labels to categorical one-hot encoding
#y_train = keras.utils.to_categorical(y_train, num_classes=10)
#y_test = keras.utils.to_categorical(y_test, num_classes=10)

checkPoint=ModelCheckpoint("cifar10.cnn", save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
default_callbacks = default_callbacks+[checkPoint]

earlyStopping=EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=10, verbose=0, mode='min') 
default_callbacks = default_callbacks+[earlyStopping]


# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, y_train, validation_split = 0.2, epochs=100, batch_size=2, callbacks = default_callbacks)

score = model.evaluate(X_test, y_test, batch_size=32)
print(score)