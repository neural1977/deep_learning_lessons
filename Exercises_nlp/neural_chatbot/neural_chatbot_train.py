'''
Created on 19/02/2020
Modified on 19/02/2020


@author: Vamsi Gunturi, Francesco Pugliese
'''


import numpy as np
import tensorflow as tf
import pickle
from keras.layers import GRU, LSTM, Embedding, Input, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.activations import softmax
from keras.optimizers import RMSprop


import requests, zipfile, io

import os
import yaml
import pdb
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
hiddent_dim = 200

# Set CPU or GPU type
gpu = True
gpu_id = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

dir_path = '../../Datasets/Chatterbot_dataset'
files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()

# Dataset preparing with questions and answers lists
for filepath in files_list:
    stream = open( dir_path + os.sep + filepath , 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len( con ) > 2 :
            questions.append(con[0])
            replies = con[ 1 : ]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append( ans )
        elif len( con )> 1:
            questions.append(con[0])
            answers.append(con[1])

answers_with_tags = list()
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        questions.pop( i )

answers = list()

for i in range( len( answers_with_tags ) ) :
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

tokenizer = Tokenizer()
tokenizer.fit_on_texts( questions + answers )
VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))
# Saving word index into a pickle file
    
# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences( questions )
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )

encoder_input_data = pad_sequences( tokenized_questions , maxlen=maxlen_questions , padding='post' )
#encoder_input_data = np.array( padded_questions )
print( encoder_input_data.shape , maxlen_questions )


# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
decoder_input_data = pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
#decoder_input_data = np.array( padded_answers )
print( decoder_input_data.shape , maxlen_answers )

with open('neural_chatbot_wi.pkl', 'wb') as handle:
    pickle.dump(tokenizer.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(maxlen_questions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(maxlen_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)

# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )

# Saving all the arrays to storage
#np.save( 'enc_in_data.npy' , encoder_input_data )
#np.save( 'dec_in_data.npy' , decoder_input_data )
#np.save( 'dec_tar_data.npy' , decoder_output_data )

# Model building
encoder_inputs = Input(shape=( None , ))
encoder_embedding = Embedding( VOCAB_SIZE, hiddent_dim , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = LSTM( hiddent_dim , return_state=True )( encoder_embedding )
#encoder_outputs , state_h , state_c = LSTM( hiddent_dim , return_state=True, recurrent_dropout=0.5 )( encoder_embedding )
#encoder_outputs , state_h , state_c = GRU( hiddent_dim , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = Input(shape=( None ,  ))
decoder_embedding = Embedding( VOCAB_SIZE, hiddent_dim , mask_zero=True) (decoder_inputs)
decoder_lstm = LSTM( hiddent_dim , return_state=True , return_sequences=True )
#decoder_lstm = LSTM( hiddent_dim , return_state=True , return_sequences=True, recurrent_dropout=0.5 )
#decoder_lstm = GRU( hiddent_dim , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = Dense( VOCAB_SIZE , activation=softmax ) 
output = decoder_dense ( decoder_outputs )

model = Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Model training

model.summary()

check_point = ModelCheckpoint('best_chatbot.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('chatbot_training_history.log')

model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=500, callbacks=[csv_logger, check_point], validation_split=0.4, verbose = 2) 
model.save('last_epoch_chatbot.hdf5') 

# Infererecing the models
def make_inference_models():
    
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=( hiddent_dim ,))
    decoder_state_input_c = Input(shape=( hiddent_dim ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model



def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        if word in tokenizer.word_index:
          tokens_list.append( tokenizer.word_index[ word ] ) 
    return pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

enc_model, dec_model = make_inference_models()

def get_response(input_text):

    #print("Get response is called from server")

    states_values = enc_model.predict(str_to_tokens(input_text))
   
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]

    return decoded_translation
    #print( decoded_translation.rsplit(' ', 1)[0] )


input_text = ''

while(input_text != 'stop'):
  input_text = input( 'Enter question : ' )
  print(get_response(input_text))


