# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Keras imports
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam 
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Program imports
from Preprocessing.preprocessing import LoadData
from Models.convnet import Lenet5
from Utils.view import View

# General imports
import warnings
import pdb
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)

# Initialization
default_callbacks = []

# Globals
labels_garbage_list = ["glass", "metal", "paper", "plastic", "trash"]

# Hyperparameter
learning_rate = 0.1
epochs = 10
limit = None
batch_size = 32
normalization = True

# Dataset parameters
cifar10_ds = True
garbage_ds = False
develop = False
dest_input_size = 128

if develop: 
    limit = 1000
    epochs = 10

if cifar10_ds: 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_size = x_train.shape[1]
    depth = x_train.shape[3]
    #pdb.set_trace()
    classes = len(np.unique(y_train))
    
if garbage_ds:
    [x_train, y_train, x_val, y_val, x_test, y_test] = LoadData.load_dataset_cv('G:/Dropbox/Datasets/Garbage Data Set', labels_garbage_list, 0.1, 0.1, True, limit, dest_input_size, True)
    input_size = x_train.shape[2]
    depth = x_train.shape[1]
    classes = len(np.unique(y_train))

if cifar10_ds and limit is not None: 
    x_train = x_train[0:limit]
    y_train = y_train[0:limit]
    x_test = x_test[0:limit]
    y_test = y_test[0:limit]
    
# Normalization
if normalization: 
    x_train = x_train / 255.0
    x_test = x_test / 255.0 

# One-hot encoding of output
if cifar10_ds or garbage_ds: 
    y_train = to_categorical(y_train, num_classes = classes)
    y_test = to_categorical(y_test, num_classes = classes)

model = Lenet5.build(depth, input_size, input_size, classes,True)
sgd = SGD(lr = learning_rate)
#sgd = Adam(lr = learning_rate)

model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])

check_point = ModelCheckpoint("best_epoch_model.hdf5", save_weights_only = True, monitor = "val_accuracy", verbose = 1, save_best_only = True, mode = "max")
default_callbacks = default_callbacks + [check_point]

earlyStopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 10, verbose = 0, mode = 'min') 
default_callbacks = default_callbacks + [earlyStopping]

csv_logger = CSVLogger('history.log')
default_callbacks = default_callbacks + [csv_logger]

history = model.fit(x_train, y_train, validation_split=0.2, epochs = epochs, batch_size = batch_size, shuffle=True, callbacks = default_callbacks)

model.save_weights('last_epoch_model.hdf5')
    
View.plot_loss(history)
View.plot_acc(history)