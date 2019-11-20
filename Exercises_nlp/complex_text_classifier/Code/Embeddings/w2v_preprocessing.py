'''
Created on 31/01/2019

@author: Francesco Pugliese
'''

import gensim
from gensim.models import KeyedVectors
import pdb

def initialize_embeddings(embeddings_path, embeddings_file, fastEmbeddingsLoad, language):    
    if language is not None: 
        if language == 'en': 
            print('\nLoading Word Embeddings (English Language Corpus)...\n')
        elif language == 'it': 
            print('\nLoading Word Embeddings (Italian Language Corpus)...\n')
    
    if fastEmbeddingsLoad == False: 
        model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path+'/'+embeddings_file, binary = False)
    else: 
        model = KeyedVectors.load(embeddings_path+'/'+embeddings_file, mmap = 'r')

    weights = model.syn0
    
    return model

def encode_fast_embeddings(embeddings_path, embeddings_slow_file): 
    print('\nConverting Original Pre-trained Word Embeddings to a fast corpus...\n')
    model_to_convert = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path+'/'+embeddings_slow_file, binary = False)
    model_to_convert.save(embeddings_path+'/'+embeddings_slow_file+'.bin')
