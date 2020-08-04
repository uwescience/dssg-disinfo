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


### II. Import data
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)
PATH = os.getenv("PATH")
CLEAN_DATA = os.getenv("CLEAN_DATA")
df = pd.read_csv(os.path.join(PATH, CLEAN_DATA))


### III. Splitting the data into training and testing
from sklearn.model_selection import train_test_split
X = df['article_headline'] + " " + df['article_text'] # including both article text and headline
y = df.label
training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(X, y, random_state = 42)

# making y into np arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
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
model.summary()


### VI. Put model together and run
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


### VII. Fitting and running the model
num_epochs = 1
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))