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
import tensorflow as tf
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
from sklearn.metrics import roc_curve, auc

#Import Model Registry
#import model_registry
#from model_registry import *
import model_arch
from model_arch import *
import get_data
from get_data import *


import params_class
params=params_class.params()

### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)


def create_multiple_model_arch(bidir_num_filters=64, dense_1_filters=10, dropout_rate=0.2, vocab_size=10000, embedding_dim=300, maxlen=681, optimizer='adam', epochs=5, embedding_path=None): #Could we replace with (?): "**copacabana"
    nlp_input=Input(shape=[None]) # Input layer for text
    meta_input=Input(shape=(22,)) # Input layer for 22 linguistic feature columns
    nlp_embeddings=Embedding(params.vocab_size, params.embedding_dim)(nlp_input)
    nlp_LSTM=LSTM(params.bidir_num_filters)(nlp_embeddings) # text embeddings LSTM
    x = Concatenate()([nlp_LSTM, meta_input]) # Merge text LSTM with linguistic features
    x = Dense(dense_1_filters, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x) # Output layer
    model=Model(inputs=[nlp_input, meta_input], outputs=[x]) # Final model
    
    # compile may not be necessary
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
    return model
    

# Register the basic model (writing into our dictionary of models)
register_model_arch("multiple", create_multiple_model_arch,
                    ["bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen",   "optimizer", "epochs", "embedding_path", "dropout_rate"])

