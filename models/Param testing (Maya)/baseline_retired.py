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

### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)


# Creation function for the basic model architecture:
def create_basic_model_arch(vocab_size, embedding_dim, max_length):
    "Creates and returns a basic model architecture."
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64)), # 64 could be a parameter?
        Dense(64, activation='relu'), # 64 and 'relu' could be params?
        Dense(1, activation='sigmoid') # 'sigmoid' could be a param?
    ])
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("basic", create_basic_model_arch,
                    ["vocab_size", "embedding_dim", "max_length"])

# ...

def build_model(vocab_size=10000, embedding_dim=300, max_length = 681, epochs=5, model_arch='basic'):
    """Builds a model using the passed parameters."""
    # (This next line could be implemented by using def build_model(**params) instead)
    params = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'max_length': max_length,
        'epochs': epochs,
        'model_arch': 'basic'
    }
    
    # ...build up other parts of the model...
    model = build_model_arch(params['model_arch'], params)
    # ...etc...
    
    # Print model layers
    print("Model summary:")
    model.summary()

    ### VI. Put model together and run
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

 

def get_data_and_split(vocab_size, max_length):
    '''
    Fetches the data and splits into train/test
    '''
                     
     # Get the paths
    DATA_PATH = os.getenv("PATH") # we need to change "PATH" to "DATA_PATH" in the ENV File 
    CLEAN_DATA = os.getenv("CLEAN_DATA")
    df = pd.read_csv(os.path.join(DATA_PATH, CLEAN_DATA))


    ### III. Splitting the data into training and testing
    X = df['article_text'] # article_text
    y = df.label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # making y into np arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Padding and tokenizing 
    tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, truncating='post')
    
    # turning to sequence 
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length)
    
    return X_train_padded, X_test_padded, y_train, y_test
                     

def compile_model(model,loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
                     
    # Print model layers
    print("Model summary:")
    model.summary()

    ### VI. Put model together and run
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
                     
def fit_and_run_model(model, vocab_size=10000, embedding_dim=300, max_length=681, epochs=5):
    
    ## Fetching data and splitting/tokenizing/padding
    (X_train_padded, X_test_padded, y_train, y_test) = get_data_and_split(vocab_size, max_length)
    
    ## VII. Fitting and running the model
    file_name = 'LSTM_model'+'_'+str(vocab_size)+'_'+str(embedding_dim)+'_'+str(max_length)+'_'+str(epochs)+'.log'
    csv_logger = CSVLogger(file_name, append=True, separator=';')
    history=model.fit(X_train_padded, y_train, epochs=epochs, validation_data=(X_test_padded, y_test), callbacks=[csv_logger])
    return history, model



'''
                     
def LSTM_model(VOCAB_SIZE = 10000, EMBEDDING_DIM = 300, MAX_LENGTH = 681, NUM_EPOCHS = 5):
    """
    input:
    -----
    VOCAB_SIZE: The number of words from corpus
    EMBEDDING_DIM: The dimension of the embedding
    MAX_LENGTH: The number of tokens that will be kept from each 
    NUM_EPOCHS: The number of times the model will be run
    
    output:
    history: The model fit output
    """
    # Get the paths
    DATA_PATH = os.getenv("PATH") # we need to change "PATH" to "DATA_PATH" in the ENV File 
    CLEAN_DATA = os.getenv("CLEAN_DATA")
    df = pd.read_csv(os.path.join(DATA_PATH, CLEAN_DATA))


    ### III. Splitting the data into training and testing
    X = df['article_text'] # article_text
    y = df.label
    
    training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(X, y, random_state = 42)

    # making y into np arrays
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    vocab_size = VOCAB_SIZE 
    embedding_dim = EMBEDDING_DIM
    max_length = MAX_LENGTH
    trunc_type='post'
    oov_tok = "<OOV>"


    ### IV. Tokenizing, padding, and truncating
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


    ### V. Network architecture/building the model
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # Print model layers
    print("Model summary:")
    model.summary()

    ### VI. Put model together and run
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### VII. Fitting and running the model
    num_epochs = NUM_EPOCHS
    file_name = 'LSTM_model'+'_'+str(VOCAB_SIZE)+'_'+str(EMBEDDING_DIM)+'_'+str(MAX_LENGTH)+'_'+str(NUM_EPOCHS)+'.log'
    csv_logger = CSVLogger(file_name, append=True, separator=';')
    history=model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final), callbacks=[csv_logger])
    
    #plot_graphs(history, 'accuracy')
    #plot_graphs(history, 'loss')
    
    return history


def plot_graphs(history, string):
    
    #history=collected model, string= 'accuracy' or whichever. 
    #It will name the png output file as history+metric
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    file_name= f'{history}'+ string + '.png'
    plt.savefig(file_name)
    return
'''