'''
Created on 19/02/2019

@author: Vamsi, Francesco Pugliese
'''

from keras.models import Sequential
from keras.layers.core import Dense

class Mlp:
    @staticmethod
    def build(input_dim, classes, summary, weightsPath=None):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=input_dim))
        model.add(Dense(classes, activation='softmax'))

        if summary==True:
            model.summary()
        
		#if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model