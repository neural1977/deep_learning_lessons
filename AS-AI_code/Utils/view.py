# -*- coding: utf-8 -*-
"""
Created on 12/05/2020

@author: Francesco Pugliese, Vamsi Gunturi
"""
import matplotlib.pyplot as plt

class View: 
    # Plot the loss function chart
    @staticmethod
    def plot_loss(history):
        plt.figure()												# generate a new window
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(loc='upper right')
        plt.savefig("loss.jpg")
        plt.show(block=False)
        plt.pause(1)
    
    @staticmethod
    # Plot the accuracy function chart
    def plot_acc(history):
        plt.figure()												# generate a new window
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend(loc='upper left')
        plt.savefig("accuracy.jpg")
        plt.show(block=False)
        plt.pause(1)