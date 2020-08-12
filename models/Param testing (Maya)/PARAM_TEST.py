def create_model(bidir_num_filters, dense_1_filters, vocab_size, embedding_dim, maxlen, optimizer):
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

'''param_grid = dict(bidir_num_filters=[32, 64, 128],
                  dense_1_filters=[range(1, 11)],
                  vocab_size=[5000,10000,15000,20000], 
                  embedding_dim=[50, 100, 300],
                  maxlen=[100, 500, 681, 1000],
                  optimizer=['adam','SGD'],
                  dropout=[0.1])'''

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

# Main settings
epochs = 20
embedding_dim = 50
maxlen = 681
output_file = 'data/output.txt'

# Bring in data
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)
# Get the paths
DATA_PATH = os.getenv("PATH") # we need to change "PATH" to "DATA_PATH" in the ENV File 
CLEAN_DATA = os.getenv("CLEAN_DATA")
df = pd.read_csv(os.path.join(DATA_PATH, CLEAN_DATA)).head(100)


### III. Splitting the data into training and testing
sentences = df['article_text'].values
y = df['label'].values

# Train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)





# Parameter grid for grid search
'''param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])'''
param_grid = dict(bidir_num_filters=[32], #, 64, 128],
                  dense_1_filters=[1],
                  vocab_size=[5000], #10000,15000,20000], 
                  embedding_dim=[50],# 100, 300],
                  maxlen= [681], #[100, 500, 681, 1000], #
                  optimizer=['adam']) #'SGD'])
model = KerasClassifier(build_fn=create_model,
                        epochs=1, batch_size=10,
                        verbose=False)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv = StratifiedKFold(n_splits=5), verbose=1, n_iter=5) #(cv = StratifiedKFold(n_splits=split_number
grid_result = grid.fit(X_train, y_train)





# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)
print(test_accuracy)

print(grid_result.best_score_)
print(grid_result.best_params_)
print(grid_result.cv_results_)

results= grid_result
score=grid.score