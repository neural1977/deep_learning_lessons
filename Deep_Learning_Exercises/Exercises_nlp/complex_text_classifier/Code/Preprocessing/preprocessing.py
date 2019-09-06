'''
Created on 31/01/2019
Modified on 31/01/2019


@author: Francesco Pugliese
'''

# Keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# DB imports
import pymysql

# Other imports
import numpy
import os
import sys
from os import listdir
from os.path import isfile, isdir, join
import pdb
from skimage.draw import circle
from skimage.draw import circle_perimeter
import math
import datetime
import pandas

import scipy.misc
import matplotlib.pyplot as plt
	
def load_datasets(sentiment_dataset_path='', validation_split = 0.2, shuffle = False, limit = None):
    print('Loading Data Sets from files CSV...')
    trainsetdir = numpy.asarray([f for f in listdir(sentiment_dataset_path)])

    data_set = []
    n_labels_0 = 0
    n_labels_1 = 0
    pos_first = False
    positive_class_available = False															# it says if there is a positive class file
    negative_class_available = False															# it says if there is a negative class file
    for k in range(trainsetdir.shape[0]):
        print ('Reading: %s' % trainsetdir[k]) 
        file = open(sentiment_dataset_path+'/'+trainsetdir[k], 'r', encoding='latin-1')
        i = 0
        for line in file:
            i=i+1 
            splittedline = line

            if 'neg' in trainsetdir[k]:
                n_labels_0 = n_labels_0 + 1
                positive_class_available = True
            elif 'pos' in trainsetdir[k]: 
                n_labels_1 = n_labels_1 + 1
                negative_class_available = True

            if 'pos' in trainsetdir[k] and k==0: 
                pos_first = True

            data_set.append(splittedline)

            if limit is not None and i>=limit:                         # reads only few rows
                break

        print ('%i lines read' % i)

    data_set = numpy.asarray(data_set) 
    # create labels
    labels_0 = numpy.repeat(0, n_labels_0).reshape(-1,1)
    labels_1 = numpy.repeat(1, n_labels_1).reshape(-1,1)
    if pos_first == True:
        labels = numpy.vstack((labels_1, labels_0))
    else:
        labels = numpy.vstack((labels_0, labels_1))
	
    # split the data into a training set and a validation set
    indices = numpy.arange(data_set.shape[0])
    if shuffle == True: 
        numpy.random.shuffle(indices)
    data_set = data_set[indices]
    if (n_labels_0 + n_labels_1) != 0:
        labels = labels[indices]
    num_validation_samples = int(round(validation_split * data_set.shape[0]))
    x_train_txt = data_set[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val_txt = data_set[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
	
    return [(x_train_txt, y_train), (x_val_txt, y_val), (data_set, labels), (num_validation_samples), (positive_class_available, negative_class_available)]

def load_sentipolc_datasets(sentiment_dataset_path='', validation_split = 0.2, shuffle = False, limit = None, remove_neutral_mixed = False):
    print('Loading SENTIPOLC Data Set...')
    trainsetdir = numpy.asarray([f for f in listdir(sentiment_dataset_path)])

    data_set = []
    n_labels_0 = 0
    label_0 = 0.0
    n_labels_1 = 0
    label_1 = 1.0
    n_labels_00 = 0
    label_00 = 0.5
    n_labels_11 = 0
    label_11 = 0.5
    for k in range(trainsetdir.shape[0]):
        print ('Reading: %s' % trainsetdir[k]) 
        file = open(sentiment_dataset_path+'/'+trainsetdir[k], 'r', encoding='latin-1')
        i = 0
        for line in file:
            i=i+1 
            splittedline = line.split(",")                                                          
            assigned_label = 0.0
  
            if (splittedline[2] == '1' and splittedline[3] == '0') or (splittedline[2] == '"1"' and splittedline[3] == '"0"'):         # positive sentiment
                n_labels_1 = n_labels_1 + 1
                assigned_label = label_1
            elif (splittedline[2] == '0' and splittedline[3] =='1') or (splittedline[2] == '"0"' and splittedline[3] =='"1"'):         # negative sentiment
                n_labels_0 = n_labels_0 + 1
                assigned_label = label_0
            elif (splittedline[2] == '0' and splittedline[3] =='0') or (splittedline[2] == '"0"' and splittedline[3] =='"0"'):	       # neutral sentiment
                n_labels_00 = n_labels_00 + 1
                assigned_label = label_00
            elif (splittedline[2] == '1' and splittedline[3] =='1') or (splittedline[2] == '"1"' and splittedline[3] =='"1"'):         # mixed sentiment
                n_labels_11 = n_labels_11 + 1                  
                assigned_label = label_11
			
            data_set.append([splittedline[8], assigned_label])

            if limit is not None and i>=limit:                         # reads only few rows
                break

        print ('%i lines read' % i)

    data_set = numpy.asarray(data_set) 

    print ('\nNumber of tweets with Positive Sentiment: %i - assigned label: %.1f' % (n_labels_1, label_1))
    print ('Number of tweets with Negative Sentiment: %i - assigned label: %.1f' % (n_labels_0, label_0))
    print ('Number of tweets with Neutral Sentiment: %i - assigned label: %.1f' % (n_labels_00, label_00))
    print ('Number of tweets with Mixed Sentiment: %i - assigned label: %.1f\n' % (n_labels_11, label_11))

    # split the data into a training set and a validation set
    indices = numpy.arange(data_set.shape[0])
    if shuffle == True: 
        numpy.random.shuffle(indices)
    data_set = data_set[indices]

    if remove_neutral_mixed == True:
        data_set = numpy.delete(data_set, numpy.where(data_set[:,1]=='0.5')[0], 0)
	
	# Extracts the columns in order to interface with the previous format
    labels = data_set[:,1]
    data_set = data_set[:,0]
	
    num_validation_samples = int(round(validation_split * data_set.shape[0]))
    x_train_txt = data_set[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val_txt = data_set[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    
    return [(x_train_txt, y_train), (x_val_txt, y_val), (data_set, labels), (num_validation_samples)]

def create_word_index(data_set, max_words = 10000, padded_sequence_length = 100, longest = False):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(max_words)
    tokenizer.fit_on_texts(data_set) 
    sequences = tokenizer.texts_to_sequences(data_set)
    word_index = tokenizer.word_index

    if longest == True: 
        maxlen = None   
    else: 
        maxlen = padded_sequence_length   

    data = pad_sequences(sequences, maxlen=maxlen)
    
    return [data, word_index]

def prepare_embedding_matrix(word_index, max_words = 10000, embeddings_model = None, embedding_size = 300):
    # prepare embedding matrix
    num_words = min(max_words, len(word_index))
    embedding_matrix = numpy.zeros((num_words+1, embedding_size))
    not_matched = 0
    for word, i in word_index.items():
        if i >= max_words:
            continue
        if word.lower() in embeddings_model.vocab:                        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embeddings_model[word.lower()]          # gets the embedding vector associated to the word            
        else:
            not_matched = not_matched + 1              # words of training set not present in word embeddings

    return [embedding_matrix, not_matched]