'''
Created on 31/01/2019

@author: Francesco Pugliese
'''

# Keras imports
from keras.preprocessing.text import text_to_word_sequence

import numpy
import pdb

def delete_empty_tweets(xs, ys):
    all_indexes = list(range(len(xs)))
    indexes_to_delete = [ind for (ind, tweet) in enumerate(xs) if len(tweet) == 0]
    indexes_to_retain = [ind for ind in all_indexes if ind not in indexes_to_delete]
    
    xs_cleaned = [x for (i, x) in enumerate(xs) if i not in indexes_to_delete]
    ys_cleaned = ys[indexes_to_retain]
    
    return (xs_cleaned, ys_cleaned)
	
def get_occurrencies(x):									# gets a list of most frequent elements in the array x
    y = numpy.bincount(x)
    ii = numpy.nonzero(y)[0]
    f = numpy.vstack((ii,y[ii])).T
    return f[f[:,1].argsort(kind='mergesort')[::-1]]		# sort frequences in descending order by the second column

def find_best_keyword_context(keywords_list, context_list, embeddings_model, n_significant_keywords, n_significant_contextes):			# gets the best keywords and best context
    # Find the best contextes
    if len(keywords_list) != 0 and len(context_list) != 0:
        keywords_list_storage = list(keywords_list)
        i = 0
        keywords_occurrencies = []
        while i < len(keywords_list_storage): 
            num_occurrencies = 0
            j = i + 1
            while j < len(keywords_list_storage):
                
                try: 
                    sim = embeddings_model.similarity(keywords_list_storage[i], keywords_list_storage[j])			# checks the similarity between couples of words
                except KeyError: 
                    sim = 0
				
                if sim > 0.6:
                    num_occurrencies = num_occurrencies + 1
                    keywords_list_storage.remove(keywords_list_storage[j])            				
                
                j = j + 1				
			
            keywords_occurrencies.append([keywords_list_storage[i], num_occurrencies])		
            i = i + 1		
        
        keywords_occurrencies = numpy.vstack(keywords_occurrencies)
        all_best_keywords = keywords_occurrencies[keywords_occurrencies[:,1].argsort(kind='mergesort')][::-1][:,0]
        n_keywords = len(all_best_keywords)
        if n_significant_keywords is not None: 
            if n_keywords > n_significant_keywords:
                n_keywords = n_significant_keywords
        best_keywords = all_best_keywords[0:n_keywords]
        context_scores = []
        for c in range(len(context_list)):
            splitted_context = text_to_word_sequence(context_list[c][0], lower = True)
            score = 0
            for word in splitted_context: 
                for k in range(len(keywords_occurrencies)): 
                    try: 
                        sim = embeddings_model.similarity(word, keywords_occurrencies[k][0])	             		# checks the similarity between couples of words
                    except KeyError: 
                        sim = 0
               
                    if sim > 0.6:
                        score = score + int(keywords_occurrencies[k][1])

            context_scores.append([score, c])		
        
        context_scores = numpy.vstack(context_scores)
        all_best_contextes = context_list[context_scores[context_scores[:,0].argsort(kind='mergesort')][::-1][:,1]]		# Extracts the best scored contextes
        n_contextes = len(all_best_contextes)
        if n_significant_contextes is not None: 
            if n_contextes > n_significant_contextes:
                n_contextes = n_significant_contextes
        best_contextes = all_best_contextes[0:n_contextes]
        
    return [best_keywords, best_contextes]

