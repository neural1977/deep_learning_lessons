'''
Created on 31/01/2019

@author: Francesco Pugliese
'''

# Keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

# Preprocessing
from Preprocessing.preprocessing import create_word_index, prepare_embedding_matrix

# Models imports
from Models.ksentiment_model_compose import SentimentModelCompose

# Utilities
from Misc.utils import get_occurrencies

import os
import numpy
import sys
import pdb


class Classification:
    @staticmethod
    def classify_tweet(tweets, mode, batch_size, embeddings_model, embedding_size, neural_model, models_path_file, max_nb_words, padded_sequence_length, context_mode, context_size, longest, padding, num_keywords):    

        symbol = "~!@#$%^&*()_-+={}[]:>;,</?*-+."

		# create tweet word index
        data, word_index = create_word_index(data_set = tweets, max_words = max_nb_words, padded_sequence_length = padded_sequence_length, longest = longest, padding = padding)                                                     # Translate textual words into indices words

        if len(data[0]) > 0:                    # if note empty tweet returns sentiment otherwise returns None
            # prepare tweet embedding magtrix
            embedding_matrix, not_matched = prepare_embedding_matrix(word_index = word_index, max_words = max_nb_words, embeddings_model = embeddings_model, embedding_size = embedding_size)

            if mode==0:
                [deepnetwork, deepnetwork_with_embeddings, deepnetwork_with_embeddings_keywords] = SentimentModelCompose.build(input_length=1, embedding_matrix=embedding_matrix, neural_model = neural_model, mode = mode, summary=False)
                if os.path.isfile(models_path_file):
                    deepnetwork.load_weights(models_path_file)     
                else:
                    print('\nPre-trained model not found: %s.' % (models_path_file))
                    sys.exit("")
                deepnetwork_with_embeddings.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                prob = deepnetwork_with_embeddings.predict(data[0], batch_size=padded_sequence_length, verbose=0)      
                sentiment = numpy.mean(prob[numpy.where(data[0]!=0)])
            elif mode==1 or mode == 2: 
                [deepnetwork, deepnetwork_with_embeddings, deepnetwork_with_embeddings_keywords] = SentimentModelCompose.build(input_length=padded_sequence_length, embedding_matrix=embedding_matrix, neural_model = neural_model, mode = mode, summary=False)   
                if os.path.isfile(models_path_file):
                    deepnetwork.load_weights(models_path_file)     
                else:
                    print('\nPre-trained model not found: %s.' % (models_path_file))
                    sys.exit("")

                sentiment = deepnetwork_with_embeddings.predict(data, batch_size=batch_size, verbose=1)		# sentiment classification

                if deepnetwork_with_embeddings_keywords is not None and num_keywords>0: 					# if there are keywords from the model and if they are required
                    
					# Keywords extraction
                    keywords_indices_list = deepnetwork_with_embeddings_keywords.predict(data, batch_size=batch_size, verbose=0)
                    
                    #print("\nFinding keywords list from keywords indices...")
				
                    keywords_list = []
                    for keywords_index in keywords_indices_list:
                        keywords_list.append(get_occurrencies(keywords_index))                                          # gets the list of most frequent words in the pool layer
				     
                    w_list_set = []
                    for dd in range(len(keywords_list)):

                        keywords = keywords_list[dd]
                        w_list = []
                        for k in range(len(keywords)):
                            kw_index = data[dd][keywords[k][0]]
                            if kw_index != 0:
                                splitted_tweet = text_to_word_sequence(tweets[dd], lower=True)
                                w_list.append([splitted_tweet[keywords[k][0] - numpy.where(data[dd] == 0)[0][-1]-1]])         # extraxts the index of the element in data an extracts the keyword directly from the tweet, it's faster than the previous one using the word index
                                #w_list.append([word for word, idx in word_index.items() if idx == kw_index])				# extracts the word associated to the found index within the word index
                        w_list_set.append(w_list)
                else:
                    w_list_set = None				
            
            if w_list_set is not None: 
                w_list_set_reduced = []
                for w_list in w_list_set:
                    if num_keywords is not None: 
                        w_list = numpy.asarray(w_list)[0:num_keywords]
                    else: 
                        w_list = numpy.asarray(w_list)
                    w_list_set_reduced.append(w_list)
                w_list_set = w_list_set_reduced	
				
									
			# determine the sentiment probabilities normalized between -1.0 and +1.0
            normalized_sentiment = 2*sentiment - 1

            sentiment_classes = []
            for ns in normalized_sentiment:
                # determine the sentiment in 5 classes
                if ns < -0.2:
                    sc = "-1"
                elif ns >= -0.2 and ns < 0:
                    sc = "-0.5"
                elif (ns >= 0 and ns < 0.2) or numpy.isnan(ns):
                    sc = "0"
                elif ns >= 0.2 and ns < 0.4:
                    sc = "+0.5"
                else:
                    sc = "+1"
                sentiment_classes.append(sc)

            sentiment_train = []
            for s in sentiment:
                # determine the sentiment for further trainings
                if s <= 0.5 or numpy.isnan(s):
                    st = "0"
                else:
                    st = "1"
                sentiment_train.append(st)

			# determine lists of keywords and lists of the contextes for each keyword 
			# Note: keywords are separated by ', ' whereas contextes are separated by ' --- ' 
            keywords_grouped_list = []
            contextes_grouped_list = [] 

            #print("\nGrouping keywords and contextes...")
			
            t = 0
            for w_list in w_list_set:								# loop in the list of lists of keywords
                keywords = None
                contextes = None
                if w_list is not None and len(w_list)!=0: 
                    keywords = w_list
                    if len(keywords[0][0]) != 0:
                        keywords_grouped = keywords[0][0]
				
                    for i in range(1, len(keywords)):
                        if len(keywords[i][0]) != 0:
                            keywords_grouped = keywords_grouped + ', ' + keywords[i][0]
										
                    contextes_grouped = ''
                    half_context_size = int((context_size - 1) / 2)
                    for i in range(0, len(keywords)):
                        splitted_tweet = text_to_word_sequence(tweets[t], lower=True)

                        if context_mode == 1:
                            try: 
                                central_index = tweets[t].index(keywords[i][0])
                                start_index = 0
                                for kk in range(central_index-1, 0, -1):
                                    if tweets[t][kk] in symbol:
                                        start_index = kk							    # start of sentence
                                        break
                                end_index = len(tweets[t])
                                for kk in range(central_index+1, len(tweets[t])):
                                    if tweets[t][kk] in symbol:
                                        end_index = kk									# end of sentence
                                        break
						
                                if start_index != 0:  
                                    start_index = start_index + 1
							
                                keyword_context = tweets[t][start_index:end_index]
                            except ValueError:
                                keyword_context = ""
                        else: 
                            keyword_context = ' '.join(splitted_tweet[splitted_tweet.index(keywords[i][0]) - half_context_size:splitted_tweet.index(keywords[i][0]) + half_context_size + 1])
         
                        if len(keyword_context) != 0:
                            if i==0: 
                                contextes_grouped = keyword_context 
                            else: 
                                contextes_grouped = contextes_grouped + ' --- ' + keyword_context
								
                t = t + 1

                keywords_grouped_list.append(keywords_grouped)
                contextes_grouped_list.append(contextes_grouped)

				
            return [normalized_sentiment, sentiment_classes, sentiment_train, keywords_grouped_list, contextes_grouped_list]
        else:
            return [None, None, None, None]