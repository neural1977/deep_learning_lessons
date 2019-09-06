'''
Created on 19/02/2019

@author: Vamsi, Francesco Pugliese
'''

# Experimantal Setup - Continuous Learning experiments on MNIST, Fashion Mnist

# general imports
import keras
import numpy as np
import pdb
import matplotlib.pyplot as plt

chunk_size = 10000
epochs = 10
batch_size = 32
number_of_train_set_pictures_to_plot = 1
number_of_test_set_pictures_to_plot = 1
plot_figures = True
save_figures = False
continuous_learning = True
#dataset_type = 'mnist'
#dataset_type = 'fashion_mnist'
dataset_type = 'cifar10'

# keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.preprocessing.image import array_to_img, img_to_array
from keras.utils import to_categorical

from Models.kconvnet import ConvNet
from Models.kmlp import Mlp

#download data and split into train and test sets
if dataset_type == 'mnist': 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    input_dim = X_train.shape[1] * X_train.shape[2]
elif dataset_type == 'fashion_mnist': 
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    input_dim = X_train.shape[1] * X_train.shape[2]
elif  dataset_type == 'cifar10': 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

if plot_figures == True: 
    for k in range(number_of_train_set_pictures_to_plot):
        # Show the first image from the training set
        if dataset_type is 'mnist' or dataset_type is 'fashion_mnist': 
            plt.imshow(X_train[k], cmap = 'gray')
        elif dataset_type is 'cifar10': 
            plt.imshow(array_to_img(X_train[k]))
        if save_figures == True:
            plt.savefig("First_fashion_mnist_train_image_"+str(k)+".jpg")
        print("First mnist train image", y_train[0])
        plt.show(block = False)
        plt.pause(2)
        plt.close()

    for k in range(number_of_test_set_pictures_to_plot):
        # Show the first image from the test set
        if dataset_type is 'mnist' or dataset_type is 'fashion_mnist': 
            plt.imshow(X_test[k], cmap = 'gray')
        elif dataset_type is 'cifar10': 
            plt.imshow(array_to_img(X_test[k]))
        if save_figures == True:
            plt.savefig("First_fashion_mnist_test_image_"+str(k)+".jpg")
        print("First mnist test image", y_test[0])
        plt.show(block = False)
        plt.pause(2)
        plt.close()

if dataset_type is 'mnist' or dataset_type is 'fashion_mnist': 
    X_train = X_train.reshape(X_train.shape[0], input_dim)
    X_test = X_test.reshape(X_test.shape[0], input_dim)

# Normalization (testare come senza normalizzazione converge molto tardi a 95% rispetto alla normalizzazione)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Split
if continuous_learning == True and (len(X_train) % chunk_size == 0): 
    X_trains = []
    y_trains = []
    for i in range(0, len(X_train) // chunk_size):
        X_trains.append(X_train[i*chunk_size:chunk_size+i*chunk_size])
        y_trains.append(y_train[i*chunk_size:chunk_size+i*chunk_size])

if dataset_type is 'mnist' or dataset_type is 'fashion_mnist': 
    model = Mlp.build(input_dim, 10, True, None)
elif dataset_type is 'cifar10': 
    model = ConvNet.build(X_train.shape[2], X_train.shape[3], X_train.shape[1], 10, True, None)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if continuous_learning == True and (len(X_train) % chunk_size == 0): 
    for i in range(0, len(X_train) // chunk_size):
        # Convert labels to categorical one-hot encoding
        one_hot_y_trains = to_categorical(y_trains[i], num_classes=10)
   
        # Train the model, iterating on the data in batches of batch_size samples
        model.fit(X_trains[i], one_hot_y_trains, validation_split = 0.2, epochs=epochs, batch_size=batch_size, verbose = 2)
else: 
    # Convert labels to categorical one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)

    # Train the model, iterating on the data in batches of batch_size samples
    model.fit(X_train, y_train, validation_split = 0.2, epochs=epochs, batch_size=batch_size, verbose = 2)

y_test = to_categorical(y_test, num_classes=10)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print(score)