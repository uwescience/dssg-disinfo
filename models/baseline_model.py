#Import Model Registry
#import model_registry
#from model_registry import *
import model_arch
from model_arch import *


### I. Importing necessary packages
import numpy as np
import pandas as pd
import io
import os
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


'''def get_basic_params():
    params = {
            'bidir_num_filters': bidir_num_filters,
            'dense_1_filters': dense_1_filters,
            'optimizer': optimizer,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'maxlen': maxlen,
            'epochs': epochs,
            'model_arch': 'basic'
    }
    return bidir_num_filters, dense_1_filters, vocab_size, embedding_dim, maxlen, optimizer'''


def create_basic_model_arch(bidir_num_filters=64, dense_1_filters=10, vocab_size=10000, embedding_dim=300, maxlen=681, optimizer='adam'):
    
    model = Sequential()
    
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(bidir_num_filters)))
    model.add(layers.Dense(dense_1_filters, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("basic", create_basic_model_arch,
                    ["bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen",   "optimizer"])





'''def create_basic_model_arch(bidir_num_filters, dense_1_filters, vocab_size, embedding_dim, maxlen, optimizer):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(bidir_num_filters)))
    #model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    #model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(dense_1_filters, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("basic", create_basic_model_arch,
                    ["bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen",   "optimizer"])'''


    
    
    
    
'''# Creation function for the basic model architecture:
def create_basic_model_arch(vocab_size, embedding_dim, maxlen):
    "Creates and returns a basic model architecture."
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(64))) # 64 could be a parameter?
    model.add(Dense(64, activation='relu')) # 64 and 'relu' could be params?
    model.add(Dense(1, activation='sigmoid')) # 'sigmoid' could be a param?
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("basic", create_basic_model_arch,
                    ["vocab_size", "embedding_dim", "maxlen"])'''