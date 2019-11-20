'''
Created on 31/01/2019

@author: Francesco Pugliese
'''

# Keras imports
from keras.callbacks import Callback

class EpochsHistory(Callback):
    def __init__(self, epochs, func, id):               # you can passa function to execute
    	self.epochs = epochs
    	self.func = func
    	self.id = id

    def on_train_begin(self, logs={}):
    	self.epoch_count = 0
    	self.epoch_perc = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count = self.epoch_count + 1
        self.epoch_perc = int(self.epoch_count * 100 / self.epochs)
        print("%i%%" % (self.epoch_perc))
        if self.func is not None: 
            self.func(self.id, self.epoch_perc)                            # update function of a db for example