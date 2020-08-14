from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import io
import os
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
from datetime import datetime
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

from word_embedding_arch import *
from get_data import *


import params_class
params=params_class.params()

import get_data
from get_data import *

#import run_model
#from run_model import run_model
import baseline_model
from baseline_model import create_basic_model_arch

def param_tune(model_arch, **copacabana):
    '''Runs parameter tuning..'''
        
    #Parameter grid for grid search
    param_grid = dict(bidir_num_filters=[32, 64, 128],
                      dense_1_filters=[10],
                      vocab_size=[10000],
                      embedding_dim=[300],
                      maxlen=[681],
                      optimizer=['adam', 'nadam'])
    
    # File to save the model logs
    file_name = datetime.now().strftime('%Y%m%d%H%M%S') +'_'+model_arch+'.log'
    csv_logger = CSVLogger(file_name, append=True, separator=';')
    
    if model_arch == 'basic':
        
        model_new = KerasClassifier(build_fn = create_basic_model_arch) # baseline_model.py function
        grid = RandomizedSearchCV(estimator=model_new, param_distributions=param_grid, cv = StratifiedKFold(n_splits=2), verbose=0, n_iter=1, scoring='accuracy')
        
        print("Loading data.")
        #pull in data
        X_train, X_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen) # get_data.py function

        grid.fit(X_train, y_train,
                 epochs=params.epochs,
                 validation_data=(X_test, y_test))
        
        print("Best model parameters are:")
        print(grid.best_params_)
        
        print(f"Fitting the best model with data. Refer {file_name}")
        
        grid.best_estimator_.fit(X_train,
                                 y_train,
                                 epochs=params.epochs,
                                 validation_data=(X_test, y_test),
                                 callbacks=[csv_logger])
        
        ### the following code saves the predictions into a file for future analysis
        predictions_file_name = datetime.now().strftime('%Y%m%d%H%M%S')+'_'+model_arch+'.predict'
        
        df1=pd.DataFrame(y_test)
        pred=grid.predict(X_test)
        df2=pd.DataFrame(pred)
        df_new=pd.concat([df1, df2],keys=['label','predicted_label'], axis=1)
        df_new.to_csv(predictions_file_name, sep=',',index=False )
        ################
        
        return file_name #file_name of epoch log of best model
    
    elif model_arch == 'multiple':
        return None, None
    
    elif model_arch == 'word_embedding':
        
        model_new = KerasClassifier(build_fn = create_word_embd_model_arch) # updated this for embd

        grid = RandomizedSearchCV(estimator=model_new, param_distributions=param_grid, 
                            cv = StratifiedKFold(n_splits=5), verbose=0, n_iter=5, scoring='accuracy')
        
        #pull in data
        X_train, X_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen)

        history = grid.fit(X_train, y_train,
                           callbacks=[csv_logger],
                           epochs=params.epochs,
                           validation_data=(X_test, y_test))
        
        test_accuracy = grid.score(X_test, y_test)
        
        print("Best model parameters are:")
        print(grid_result.best_params_)

        return history, grid.best_estimator_

    else:
        return None, None