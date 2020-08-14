import model_arch
from model_arch import *

from tensorflow import keras
from keras.layers.embeddings import Embedding
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Bidirectional, Dropout


def create_basic_model_arch(bidir_num_filters=64, dense_1_filters=10, vocab_size=10000, embedding_dim=300, maxlen=681, dropout_rate=0.2, optimizer='adam'):
    
    model = Sequential()
    
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(bidir_num_filters)))
    model.add(layers.Dense(dense_1_filters, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
    
    return model


# Register the basic model (writing into our dictionary of models)
register_model_arch("basic", create_basic_model_arch,
                    ["bidir_num_filters", "dense_1_filters", "vocab_size", "embedding_dim", "maxlen", "dropout_rate", "optimizer"])