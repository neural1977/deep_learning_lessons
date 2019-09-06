'''
Created on 27/04/2017
Modified on 12/07/2017


@author: Francesco Pugliese, Matteo Testi
'''
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from keras.models import Sequential
import pdb

class SentimentModelConvLSTM:
    @staticmethod
    def build(input_length, vector_dim):
      
        # initialize the model
        deepnetwork = Sequential()
        
        deepnetwork.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', padding = 'same', input_shape = (input_length, vector_dim)))
        deepnetwork.add(MaxPooling1D(pool_size = 5))
        deepnetwork.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        deepnetwork.add(Dense(1, activation = 'sigmoid'))

        return deepnetwork

