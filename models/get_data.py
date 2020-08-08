#--------------------------------
#       
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


#Import Model Registry
#import model_registry
#from model_registry import *
import model_arch
from model_arch import *

def get_data_and_split(vocab_size, maxlen):
    '''
    Fetches the data and splits into train/test
    '''
                     
     # Get the paths
    DATA_PATH = os.getenv("PATH") # we need to change "PATH" to "DATA_PATH" in the ENV File 
    CLEAN_DATA = os.getenv("CLEAN_DATA")
    df = pd.read_csv(os.path.join(DATA_PATH, CLEAN_DATA))
    
    ### III. Splitting the data into training and testing
    sentences = df['article_text'].values
    y = df['label'].values

    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state = 42)
    
    # making y into np arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    
    # Tokenize words
    tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen, truncating='post')
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen, truncating='post')
    
    return X_train, X_test, y_train, y_test