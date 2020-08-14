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
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Embedding, Dense, LSTM, Bidirectional,
                                    Dropout, Activation, Concatenate,
                                    Flatten, Conv1D, MaxPooling1D)
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import (Input, Model, layers)
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Import Model Registry
#import model_registry
#from model_registry import *
import model_arch
from model_arch import *
import get_data
from get_data import *

### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)


def create_multiple_model_arch(bidir_num_filters=64, dense_1_filters=10, vocab_size=10000, embedding_dim=300, maxlen=681, optimizer='adam'): #Could we replace with (?): "**copacabana"
    
    # define two different sets of inputs
    nlp_input = Input(shape=(None)) # Input layer for text
    meta_input = Input(shape=(,22)) # Input layer for 22 linguistic feature columns
    
    # define two different sets of inputs
    ###nlp_input = Input(shape=(vocab_size,embedding_dim)) # Input layer for text
    ###meta_input = Input(shape=(vocab_size,22)) # Input layer for 22 linguistic feature columns
    
    # BRANCH ONE: nlp_input layer
    x = Embedding(vocab_size, embedding_dim)(nlp_input)
    x = LSTM(bidir_num_filters)(x)# text embeddings LSTM
    x = Model(inputs=nlp_input, outputs=x)
    
    # BRANCH TWO: meta_input layer
    y = Dense(2, activation="relu")(meta_input) #Maya added this layer
    y = Model(inputs=meta_input, outputs=y)
    
    
    combi_input = Input((3,)) # (None, 3)
    a_input = Lambda(lambda x: tf.expand_dims(x[:,0],-1))(combi_input) # (None, 1) 
    b_input = Lambda(lambda x: tf.expand_dims(x[:,1],-1))(combi_input) # (None, 1)
    
    # combine the output of the two branches
    combined_output = Concatenate(axis=1)([x.output, y.output])
    combined_input = Concatenate(axis=1)([x.input, y.input])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(2, activation="relu")(combined_output)
    z = Dense(1, activation='sigmoid')(z) # Output layer
    
    # Final model
    model = Model(inputs=combined_input, outputs=z)
    
    # compile may not be necessary
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("multiple", create_multiple_model_arch,
                    ["bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen", "optimizer"])