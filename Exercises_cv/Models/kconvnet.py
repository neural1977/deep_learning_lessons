'''
Created on 19/02/2019

@author: Vamsi, Francesco Pugliese
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

class ConvNet:
    @staticmethod
    def build(width, height, depth, classes, summary, weightsPath=None):
        # Activation Functions
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(depth, height, width)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        if summary==True:
            model.summary()
        
		#if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model