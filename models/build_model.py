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

#WHERE BASELINE MODEL WAS
# ...

def build_model(bidir_num_filters=64, dense_1_filters=10, optimizer='adam', vocab_size=10000, embedding_dim=300, maxlen = 681, epochs=5, model_arch='basic', embedding_path=None):
    """Builds a model using the passed parameters."""
    
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

        # ...build up other parts of the model...
        model = build_model_arch(params['model_arch'], params)
        # ...etc...
        
    elif model_arch == 'multiple':
        nlp_input=Input(shape=[None]) # Input layer for text
        meta_input=Input(shape=(22,)) # Input layer for 22 linguistic feature columns
        nlp_embeddings=Embedding(vocab_size, embedding_dim)(nlp_input)
        nlp_LSTM=LSTM(bidir_num_filters)(nlp_embeddings) # text embeddings LSTM
        x = Concatenate()([nlp_LSTM, meta_input]) # Merge text LSTM with linguistic features
        x = Dense(1, activation='sigmoid')(x) # Output layer
        model=Model(inputs=[nlp_input, meta_input], outputs=[x]) # Final model
        
    else:
        print("Wrong model architecture!")
        
    return model

def fit_and_run_embedding_model(bidir_num_filters=64, dense_1_filters=10, vocab_size=10000, embedding_path=None, embedding_dim=300, maxlen=681, epochs=10, model_arch=model_arch):
        # Get the paths
        DATA_PATH = os.getenv("DATA_PATH")
        ALL_FEATURES_DATA = os.getenv("ALL_FEATURES_DATA")
        df = pd.read_csv(os.path.join(DATA_PATH, ALL_FEATURES_DATA))
        ### III. Splitting the data into training and testing
        X = df['article_text'] # article_text
        y = df.label

        training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(X, y, random_state = 42)

        # making y into np arrays
        training_labels_final = np.array(training_labels)
        testing_labels_final = np.array(testing_labels)
        trunc_type='post'
        oov_tok = "<OOV>"

        ### IV. Tokenizing, padding, and truncating
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(training_sentences)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(training_sentences)
        padded = pad_sequences(sequences,maxlen=maxlen, truncating=trunc_type)
        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences,maxlen=maxlen)
        
        embedding_matrix=create_embedding_matrix(embedding_path,
                                        word_index, embedding_dim)
        vocab_size = len(word_index) + 1 # Adding again 1 because of reserved 0 index
        model = keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=maxlen),
            keras.layers.Bidirectional(keras.layers.LSTM(bidir_num_filters)),
            keras.layers.Dense(dense_1_filters, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Print model layers
        print("Model summary:")
        model.summary()

        ### VI. Put model together and run
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        ### VII. Fitting and running the model
        num_epochs = epochs
        file_name = datetime.now().strftime('%Y%m%d%H%M%S')+'_'+model_arch+'_'+str(vocab_size)+'_'+str(embedding_dim)+'_'+str(maxlen)+'_'+str(epochs)+'.log'
        csv_logger = CSVLogger(file_name, append=True, separator=';')
        
        history=model.fit(padded, training_labels_final,
                          epochs=num_epochs,
                          validation_data=(testing_padded, testing_labels_final),
                         callbacks=[csv_logger])
        
        return history, model


def compile_model(model, optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']):
    """ compile model
    """
    # Print model layers
    print("Model summary:")
    model.summary()

    ### VI. Put model together and run
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model

                     
def fit_and_run_model(model, vocab_size=10000, embedding_dim=300, maxlen=681, epochs=5, model_arch='basic'):
    
    file_name = datetime.now().strftime('%Y%m%d%H%M%S') +'_'+model_arch+'_'+str(vocab_size)+'_'+str(embedding_dim)+'_'+str(maxlen)+'_'+str(epochs)+'.log'
    csv_logger = CSVLogger(file_name, append=True, separator=';')
    
    if model_arch == 'basic':
        
        ## Fetching data and splitting/tokenizing/padding
        (X_train, X_test, y_train, y_test) = get_data_and_split(vocab_size, maxlen)
        
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[csv_logger])
        
    elif model_arch == 'multiple':
        
        nlp_data_train, nlp_data_test, meta_data_train, meta_data_test, y_train, y_test = get_data_and_split(vocab_size, maxlen, multiple=True)
        history = model.fit([[nlp_data_train, meta_data_train]], y_train, 
                            epochs =epochs,
                            validation_data = (nlp_data_test, meta_data_test, y_test),
                           callbacks=[csv_logger])
    else:
        
        print("Wrong model architecture!")
        
    return history, model

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            l = len(line.split())
            word = line.split()[0]
            vector = line.split()[-embedding_dim:]
            if word in word_index:
                #n = n + 1
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

""" This is commented because get_data_and_split is now being called from get_data.py module. This is double.

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

"""



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