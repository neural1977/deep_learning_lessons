'''
Created on 27/04/2017
Modified on 12/07/2017


@author: Francesco Pugliese, Matteo Testi
'''
from keras.layers import Dense, Flatten
from keras.models import Sequential
import pdb

class SentimentModelShallow:
    @staticmethod
    def build(input_length, vector_dim):
      
        # initialize the model
        deepnetwork = Sequential()
        
        deepnetwork.add(Flatten(input_shape = (input_length, vector_dim)))
        deepnetwork.add(Dense(128, activation = 'relu'))
        deepnetwork.add(Dense(1, activation = 'sigmoid'))

        return deepnetwork

