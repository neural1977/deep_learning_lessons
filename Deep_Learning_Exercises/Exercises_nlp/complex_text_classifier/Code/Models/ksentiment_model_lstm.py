'''
Created on 27/04/2017

@author: Francesco Pugliese, Matteo Testi
'''
from keras.layers import LSTM, Dense
from keras.models import Sequential
import pdb

class SentimentModelLSTM:
    @staticmethod
    def build(input_length, vector_dim):
      
        # initialize the model
        deepnetwork = Sequential()
        
        deepnetwork.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True, input_shape = (input_length, vector_dim)))   # return_sequences to stack more LSTM
        deepnetwork.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape = (input_length, vector_dim)))
        deepnetwork.add(Dense(1, activation = 'sigmoid'))

        return deepnetwork

