# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam 
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from Models.convnet import Lenet5
import warnings
import pdb
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)

# Hyperparameter
learning_rate = 0.1
epochs = 2
save_best_model = False
limit = None
batch_size = 32
normalization = True
cifar10_ds = True
develop = False

if cifar10_ds == True: 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_size = x_train.shape[2]
    depth = x_train.shape[1]
    classes = len(np.unique(y_train))
    
if develop == True: 
    limit = 1000
    epochs = 2

if limit is not None: 
    x_train = x_train[0:limit]
    y_train = y_train[0:limit]
    x_test = x_test[0:limit]
    y_test = y_test[0:limit]
    
#pdb.set_trace()
    
# Normalization
if normalization == True: 
    x_train = x_train / 255.0
    x_test = x_test / 255.0 

# One-hot encoding of output
if cifar10_ds == True: 
    y_train = to_categorical(y_train, num_classes = classes)
    y_test = to_categorical(y_test, num_classes = classes)

model = Lenet5.build(depth, input_size, input_size, classes,True)
sgd = SGD(learning_rate = learning_rate)
#sgd = Adam(lr = learning_rate)
#ccentropy = categorical_crossentropy(from_logits = False) 


model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])

model.fit(x_train, y_train, validation_split=0.2, epochs = epochs, batch_size = batch_size, shuffle=True)

if save_best_model == False: 
    model.save_weights('last_epoch_model.hdf5')
    
# Prediction Stage
print(x_test[10,:])
print(y_test[10])

pred = model.predict(x_test)

print(pred.shape)
print("Prediction probs: ", pred[10])
print("Sum of probs: ", pred[10].sum())
print("Prediction class: ", np.argmax(pred[10]))

print(x_test[10,:].shape)
num_image = 10
num_elements= 200
print("Label: ", y_test[num_image])
x_test_image1= x_test[10,:]
x_test_image2= x_test[10:10+num_elements,:]
#pred_class = model.predict_classes(x_test[num_image:num_image+1,:])
pred_class = model.predict_classes(x_test_image2, batch_size = 200)
print(x_test_image1.shape)
print(x_test_image2.shape)
print(pred_class.shape)
print("Prediction class: ", pred_class[0])