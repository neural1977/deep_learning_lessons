'''
Created on 24/04/2017
Modified on 12/07/2017


@author: Andrea Mercuri, Francesco Pugliese, Matteo Testi
'''

from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Lambda, Dense
from keras.layers.merge import concatenate
from keras import backend

import pdb


class SentimentModelRecConvNet:
    @staticmethod
    
    def build(input_length, vector_dim):
        hidden_dim_RNN = 200
        hidden_dim_Dense = 100        
        
        embedding = Input(shape=(input_length, vector_dim))
        
        left_context = LSTM(hidden_dim_RNN, return_sequences = True)(embedding)												# Equation 1
        # left_contex: batch_size x tweet_length x hidden_state_dim
        right_context = LSTM(hidden_dim_RNN, return_sequences = True, go_backwards = True)(embedding)						# Equation 2
        # right_cntext: come left_contex
        together = concatenate([left_context, embedding, right_context], axis = 2)											# Equation 3
        semantic = TimeDistributed(Dense(hidden_dim_Dense, activation = "tanh"))(together)									# Equation 4	
        pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_Dense, ))(semantic) 				# Equation 5
        pool_rnn_args = Lambda(lambda x: backend.argmax(x, axis=1), output_shape = (hidden_dim_Dense, ))(semantic)
		
        output = Dense(1, input_dim = hidden_dim_Dense, activation = "sigmoid")(pool_rnn) 			       					# Equations 6, 7

        deepnetwork = Model(inputs=embedding, outputs=output)
        deepnetwork_keywords = Model(inputs=embedding, outputs=pool_rnn_args)

        return [deepnetwork, deepnetwork_keywords]
  