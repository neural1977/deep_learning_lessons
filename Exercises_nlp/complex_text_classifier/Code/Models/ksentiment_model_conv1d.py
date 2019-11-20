'''
Created on 27/04/2017
Modified on 12/07/2017


@author: Francesco Pugliese, Matteo Testi
'''
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from keras.models import Sequential

class SentimentModelConv1D:
    @staticmethod
    def build(input_length, vector_dim):
      
        # initialize the model
        deepnetwork = Sequential()
        
        deepnetwork.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', padding = 'same', input_shape = (input_length, vector_dim)))
        deepnetwork.add(MaxPooling1D(pool_size = 5))
        deepnetwork.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', padding = 'same'))
        deepnetwork.add(MaxPooling1D(pool_size = 5))
        #deepnetwork.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', padding = 'same'))
        #deepnetwork.add(MaxPooling1D(pool_size = 5))
        deepnetwork.add(Flatten())
        deepnetwork.add(Dense(128, activation = 'relu'))
        deepnetwork.add(Dense(1, activation = 'sigmoid'))

        return deepnetwork

