'''
Created on 26/02/2020

@author: Francesco Pugliese
'''

from __future__ import print_function
import pdb

# Keras imports
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing
from Preprocessing.preprocessing import load_datasets, load_sentipolc_datasets, create_word_index, prepare_embedding_matrix
from Embeddings.w2v_preprocessing import initialize_embeddings, encode_fast_embeddings

# Training
from Training.training import Training

# Misc
from Misc.utils import delete_empty_tweets

# Other imports
import numpy
from os import listdir
import random
import os
import timeit
import platform
import string
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Operating System
OS = platform.system()                      # returns 'Windows', 'Linux', etc
language = 'en'

# preprocessing
config_path = "../Config"
fastEmbeddingsLoad = True
fastEmbeddingsSave = False

# train set
MAX_NB_WORDS = 10000
validation_split = 0.02
trainset_limit = 200
#trainset_limit = None

# Training
batch_size = 32
epochs = 10
save_best_model = True
save_log = False
return_epochs = True
solver = 'adadelta'
incremental = False

# Embeddings
embedding_size = 300
embeddings_type = 'word2vec'						# Embeddings type: w2v = word2vec, glove = Glove

# Initialization 

import tensorflow as tf
from keras import backend as K

# Set CPU or GPU type
gpu = True
gpu_id = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

sentiment_dataset_path = 'G:/Sentiment_Analysis_Data/'+language+'/trainsets/'

padded_sequence_length = 140
longest = False                      # True = considering the maximum twitters length

if fastEmbeddingsLoad == False: 
    embeddings_file = 'wiki.'+language+'.vec'
else: 
    embeddings_file = 'wiki.'+language+'.vec'+'.bin'

if fastEmbeddingsSave == True:
    embeddings_slow_file = 'wiki.'+language+'.vec'

embeddings_path = 'G:/Sentiment_Analysis_Data/'+language+'/embeddings/'+embeddings_type+'/'+str(embedding_size)

# Model 
neural_model = 'recconvnet'
models_path = '../SavedModels/'+language
model_file = "best_sentiment_deep_model_windows.snn"        
model_file = neural_model + '_' + model_file

# Testing
save_valid_set = False		
sentiment_testset_path = 'G:/Sentiment Data/'+language+'/testsets'
       
def train_all(models_path, model_file, x_train_txt, y_train, x_val_txt, y_val, data_set, labels, num_validation_samples, save_valid_set, embeddings_model, embedding_size, save_best_model, max_nb_words):
       
    if save_valid_set == True: 
        numpy.savetxt(sentiment_testset_path+'/'+"x_validation.csv", x_val_txt, delimiter = ',', fmt = "%s")
        numpy.savetxt(sentiment_testset_path+'/'+"y_validation.csv", y_val, delimiter = ',')

    # create word index
    print('\nCreating Word Index...')
	
    data, word_index = create_word_index(data_set = data_set, max_words = max_nb_words, padded_sequence_length = padded_sequence_length, longest = longest)                                             # Translate textual words into indices words
        
    print('\nFound %s unique words within the train set.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:-num_validation_samples]
    x_val = data[-num_validation_samples:]

    # prepare embedding magtrix
    embedding_matrix, not_matched = prepare_embedding_matrix(word_index = word_index, max_words = max_nb_words, embeddings_model = embeddings_model, embedding_size = embedding_size)
    print('\n%i train-set words not found in word embeddings corpus.' % (not_matched))

    
    # Start training
    print('Building model...')

    train_start_time = timeit.default_timer()
        
    Training.train(x_train, y_train, x_val, y_val, neural_model, batch_size, epochs, embedding_matrix, models_path, model_file, save_best_model, solver, save_log, return_epochs, incremental, None, None)

    train_end_time = timeit.default_timer()

    return train_end_time - train_start_time

# Body of the program 
numpy.random.seed(23455) 
global_start_time = timeit.default_timer()


# Preprocessing
if language == "en": 
    datasets = load_datasets(sentiment_dataset_path = sentiment_dataset_path, validation_split = validation_split, shuffle = True, limit = trainset_limit)      # Loads Trainig and Validation Set
elif language == "it": 
    datasets = load_sentipolc_datasets(sentiment_dataset_path = sentiment_dataset_path, validation_split = validation_split, shuffle = True, limit = trainset_limit)      # Loads Trainig and Validation Set
else: 
    print("\nLanguage not valid: %s" % (language))
    sys.exit("")
	
x_train_txt, y_train = datasets[0] 
x_val_txt, y_val = datasets[1]
data_set, labels = datasets[2]
num_validation_samples = datasets[3]

# Word embeddings
if fastEmbeddingsSave == True:
    encode_fast_embeddings(embeddings_path, embeddings_slow_file)

embeddings_model = initialize_embeddings(embeddings_path, embeddings_file, fastEmbeddingsLoad, language)

train_time = train_all(models_path, model_file, x_train_txt, y_train, x_val_txt, y_val, data_set, labels, num_validation_samples, save_valid_set, embeddings_model, embedding_size, save_best_model, max_nb_words = MAX_NB_WORDS)
					   
print ('\n\nTraining time: %.2f minutes\n' % (train_time  / 60.))

end_time = timeit.default_timer()
print ('\n\nGlobal time: %.2f minutes\n' % ((end_time - global_start_time) / 60.))
