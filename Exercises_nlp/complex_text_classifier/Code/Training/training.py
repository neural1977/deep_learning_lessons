# Keras imports
from keras.callbacks import ModelCheckpoint, CSVLogger

# Models imports
from Models.ksentiment_model_compose import SentimentModelCompose
#from Models.ksentiment_model_rec_conv_net import SentimentModelRecConvNet

# Callbacks
from Callbacks.epochs_history import EpochsHistory
import sys
import pdb

#import tensorflow as tf

import numpy

class Training:
    @staticmethod
    def train(x_train, y_train, x_val, y_val, neural_model, batch_size, epochs, embedding_matrix, models_path, model_file, save_best_model, solver, save_log, return_epochs, incremental, func, id):

        [deepnetwork, deepnetwork_with_embeddings] = Training.setup_model(input_length=x_train.shape[1], embedding_matrix=embedding_matrix, neural_model = neural_model, summary=True, models_path=models_path, model_file=model_file, optimizer=solver, incremental=incremental)
        
        default_callbacks = []
            
        if save_log == True:
            csvLogger = CSVLogger('../Log/training.log')
            default_callbacks = default_callbacks+[csvLogger]

        if return_epochs == True:
            history = EpochsHistory(epochs, func, id)
            default_callbacks = default_callbacks+[history]

        if save_best_model == True:
            checkPoint=ModelCheckpoint(models_path+'/'+model_file, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            default_callbacks = default_callbacks+[checkPoint]
		
        deepnetwork_with_embeddings.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks = default_callbacks, validation_data=(x_val, y_val), verbose = 2)

        if save_best_model == False:                                    # we save the part of the model without the embedding layer which changes every time
            deepnetwork.save_weights(models_path+'/'+model_file)
    
    @staticmethod
    def setup_model(input_length, embedding_matrix, neural_model, summary, models_path, model_file, optimizer, incremental):
        # we do not use "deepnetwork_with_embeddings_keywords" for the training phase
        [deepnetwork, deepnetwork_with_embeddings, deepnetwork_with_embeddings_keywords] = SentimentModelCompose.build(input_length=input_length, embedding_matrix=embedding_matrix, neural_model = neural_model, summary=summary)   
    
        if incremental == True:                                         # type of training incremental or basic
            if os.path.isfile(models_path+'/'+model_file):
                deepnetwork.load_weights(models_path+'/'+model_file)     
            else:
                print('\nIt is needed a pre-trained model for Incremental Training. %s does not exist.' % (model_file))
                sys.exit("")
                    
        deepnetwork_with_embeddings.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return [deepnetwork, deepnetwork_with_embeddings]