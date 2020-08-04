#--------------------------------
#       Baseline model 
#--------------------------------

### I. Importing necessary packages
import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    return

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
    PATH = os.getenv("PATH")
    CLEAN_DATA = os.getenv("CLEAN_DATA")
    df = pd.read_csv(os.path.join(PATH, CLEAN_DATA))


    ### III. Splitting the data into training and testing
    X = df['article_headline'] + " " + df['article_text'] # including both article text and headline
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
    history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
    
    plot_graphs(history, 'acc')
    plot_graphs(history, 'loss')
    
    return history

