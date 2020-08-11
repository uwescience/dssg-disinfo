#--------------------------------
#       Model architecture file
#--------------------------------

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

### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)

#----------------------------------

#----------------------------------

### Model Registry? 

_model_arch_registry = {}

def get_model_params(model_arch='basic'):
    
    
      if(model_arch=='basic'):
        # (This next line could be implemented by using def build_model(**params) instead)
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

def register_model_arch(arch_name, create_fn, param_names):
    """
    register_model_arch(name, fn, param_names) registers a model architecture
    with the given name so that it can be built with the build_model() function.

    The name should be a string that identifies the model.
    The fn should be a function that accepts the parameters in the param_names and
        yields a sequential model.
    The param_names should be a list of the names of the parameters that fn requires.
    """
    # Save the model
    _model_arch_registry[arch_name] = (create_fn, param_names)

def build_model_arch(arch_name, param_dict):
    """
    Builds a model using the given architecture name and the given parameter
    dictionary.
    """
    # lookup the params and create function:
    (create_fn, params) = _model_arch_registry[arch_name]
    # The f(*[...]) syntax means that instead of being called as f([a,b,c]) the
    # functiion call is converted into f(a, b, c).
    return create_fn(*[param_dict[k] for k in params])


        
        