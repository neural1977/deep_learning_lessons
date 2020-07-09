# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:22:44 2020

@author: francesco
"""
from keras.datasets import cifar10
from Models.convnet import Lenet5

# Hyperparameter
learning_rate = 0.1
batch_size = 32
normalization = True
cifar10_ds = True
best_model = True

if cifar10_ds == True: 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_size = x_test.shape[1]
    depth = x_test.shape[3]
    classes = len(np.unique(y_test))
    
# Normalization
if normalization == True: 
    x_test = x_test / 255.0 

model = Lenet5.build(depth, input_size, input_size, classes, False)
model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

if best_model == True: 
    model.load_weights("best_epoch_model.hdf5")
else:     
    model.load_weights("last_epoch_model.hdf5")

if cifar10_ds == True: 
    y_test = to_categorical(y_test, num_classes = classes)

# Evaluation Stage
score = model.evaluate(x_test, y_test, batch_size=32)
print("Accuracy on test set: ", score[1]*100)

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