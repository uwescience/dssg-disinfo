#--------------------------------
# Get data and split for train and test
#--------------------------------

### I. Importing necessary packages
import numpy as np
import pandas as pd
import io
import os
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'
load_dotenv(env_path, override=True)

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #NEW
import matplotlib.pyplot as plt

#Import Model Registry
#import model_registry
#from model_registry import *
import model_arch
from model_arch import *

def get_data_and_split(vocab_size, maxlen, model_arch=None, multiple=False, scaler=False):
    '''
    Fetches the data and splits into train/test
    '''
    # Get the paths
    DATA_PATH = os.getenv("DATA_PATH") # we need to change "PATH" to "DATA_PATH" in the ENV File 
    ALL_FEATURES_DATA = os.getenv("ALL_FEATURES_DATA")
    df = pd.read_csv(os.path.join(DATA_PATH, ALL_FEATURES_DATA))
    
    ### III. Splitting the data into training and testing COULD USE METDATA ONE HERE
    
    y = df['label'].values
   
    if multiple==True:
        
        # Train-test split for two-input model (article text and metadata)
        
        train_nlp_data=df['article_text'].values
        
        train_meta_data = df[['PROPN','ADP','NOUN','PUNCT','SYM',
              'DET','CCONJ','VERB','NUM','ADV',
              'ADJ','AUX','SPACE','X','PRON',
              'PART','INTJ','SCONJ','sent_count','ratio_stops_tokens',
              'len_first_caps','len_all_caps']].values
        
        
        
        sentences_train, sentences_test, meta_data_train, meta_data_test, y_train, y_test = train_test_split(
            train_nlp_data, train_meta_data, y, test_size=0.25, random_state = 42)
        
        if scaler==True:
        # scaling metadata features to train data mean/s.d.
            scaler=preprocessing.StandardScaler().fit(meta_data_train)
            scaler.transform(meta_data_train)
            scaler.transform(meta_data_test)
        
        else:
            pass
         
        
    else:     
    # Train-test split for single input model
    
        sentences = df['article_text'].values
   
    
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state = 42)
    
    # scale 
    
    # making y into np arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Adding 1 because of reserved 0 index
    #vocab_size = len(tokenizer.word_index) + 1
    
    # Tokenize words
    tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences_train)
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

        

    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen, truncating='post')
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen, truncating='post')
    
    if multiple==True:
        return X_train, X_test, meta_data_train, meta_data_test, y_train, y_test
    
    elif model_arch=='word_embedding':
        return X_train, X_test, y_train, y_test, word_index
    
    else:
        return X_train, X_test, y_train, y_test
   
    

        


