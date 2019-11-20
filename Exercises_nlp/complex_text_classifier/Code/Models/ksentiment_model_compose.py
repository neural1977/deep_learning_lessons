'''
Created on 27/04/2017
Modified on 12/07/2017


@author: Francesco Pugliese, Matteo Testi
'''
# Keras imports
from keras.layers import Input, Embedding, ZeroPadding1D
from keras.models import Sequential, Model

# Models imports

from Models.ksentiment_model_lstm import SentimentModelLSTM
from Models.ksentiment_model_conv1d import SentimentModelConv1D
from Models.ksentiment_model_conv_lstm import SentimentModelConvLSTM
from Models.ksentiment_model_shallow import SentimentModelShallow
from Models.ksentiment_model_rec_conv_net import SentimentModelRecConvNet

import pdb
import sys

class SentimentModelCompose:
    @staticmethod
    def build(input_length, embedding_matrix, neural_model, summary):                 # compose embedding layer with model 
        embedding = Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1], weights=[embedding_matrix], input_length=input_length, trainable=False)
	
        # Calls the specific model according to the mode
        if neural_model == 'recconvnet':
            [deepnetwork, deepnetwork_keywords] = SentimentModelRecConvNet.build(input_length=input_length, vector_dim = embedding_matrix.shape[1])
        elif neural_model == 'convlstm':
            deepnetwork = SentimentModelConvLSTM.build(input_length=input_length, vector_dim = embedding_matrix.shape[1])   
        elif neural_model == 'lstm':
            deepnetwork = SentimentModelLSTM.build(input_length=input_length, vector_dim = embedding_matrix.shape[1])   
        elif neural_model == 'conv1d':
            deepnetwork = SentimentModelConv1D.build(input_length=input_length, vector_dim = embedding_matrix.shape[1])   
        elif neural_model == 'shallow':
            deepnetwork = SentimentModelShallow.build(input_length=input_length, vector_dim = embedding_matrix.shape[1])   
        else:
            print('\n %sNeural Model Unknown: ' % (neural_model))
            sys.exit("")
                
        # Build embedding layer
        deepnetwork_with_embeddings = Sequential()
        deepnetwork_with_embeddings.add(embedding)
        deepnetwork_with_embeddings.add(deepnetwork)
		
        if neural_model == 'recconvnet':
            deepnetwork_with_embeddings_keywords = Sequential()
            deepnetwork_with_embeddings_keywords.add(embedding)
            deepnetwork_with_embeddings_keywords.add(deepnetwork_keywords)
        else: 
            deepnetwork_with_embeddings_keywords = None
		
        if summary==True:
            deepnetwork_with_embeddings.summary()
            deepnetwork.summary()
			
        return [deepnetwork, deepnetwork_with_embeddings, deepnetwork_with_embeddings_keywords]