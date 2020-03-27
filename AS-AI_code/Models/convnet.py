# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:42:58 2020

@author: francesco
"""

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout

class Lenet5:
    @staticmethod
    def build(depth, height, width, classes, summary):
        model = Sequential()
        model.add(Conv2D(20, (5,5), padding = 'valid', input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(50, (5,5), padding = 'valid'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Flatten())
        #model.add(GlobalAveragePooling2D())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
    
        if summary == True:
            model.summary()
                
        return model         