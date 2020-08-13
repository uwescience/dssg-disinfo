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


import params_class
params=params_class.params()

import get_data
from get_data import get_data_and_split

#import run_model
#from run_model import run_model
import baseline_model
from baseline_model import create_basic_model_arch

import multiple_model
from multiple_model import create_multiple_model_arch

def param_tune(model_arch):
    '''Runs parameter tuning..'''
        
    #Parameter grid for grid search
    param_grid = dict(bidir_num_filters=[32, 64, 128],
                      dense_1_filters=[10],
                      vocab_size=[10000],
                      embedding_dim=[300],
                      maxlen=[681],
                      optimizer=['adam','nadam'])
    
    # File to save the model logs
    file_name = datetime.now().strftime('%Y%m%d%H%M%S') +'_'+model_arch+'.log'
    csv_logger = CSVLogger(file_name, append=True, separator=';')
    
    if model_arch == 'basic':
        
        model_new = KerasClassifier(build_fn = create_basic_model_arch) # argument is different architecture building function

        grid = RandomizedSearchCV(estimator=model_new, param_distributions=param_grid, 
                            cv = StratifiedKFold(n_splits=5), verbose=1, n_iter=5, scoring='accuracy')
        
        #pull in data
        X_train, X_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen)

        history = grid.fit(X_train, y_train,
                           callbacks=[csv_logger],
                           epochs=params.epochs,
                           validation_data=(X_test, y_test))
        
        test_accuracy = grid.score(X_test, y_test)

        return history, grid.best_estimator_
    
    elif model_arch == 'multiple':
        #model_new = KerasClassifier(build_fn = create_multiple_model_arch) # argument is different architecture building function

        #grid = RandomizedSearchCV(estimator=model_new, param_distributions = param_grid, 
        #                    cv = StratifiedKFold(n_splits=5), verbose=1, n_iter=5, scoring='accuracy', n_jobs=1)
        
        #pull in data
        ##X_train, X_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen)
        #nlp_data_train, nlp_data_test, meta_data_train, meta_data_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen, multiple=True)
        
        ## merge inputs
        #combi_train = np.concatenate((nlp_data_train, meta_data_train), axis=1)
        #combi_test = np.concatenate((nlp_data_test, meta_data_test), axis=1)
        
        #history = grid.fit(combi_train, y_train,
         #                  callbacks=[csv_logger],
          #                 epochs=params.epochs,
           #                validation_data=(combi_test, y_test))
        
        #test_accuracy = grid.score(combi_input, y_test)
        
        #print("The best parameters are:")
        #print(history.best_params_)

        #return history, history.best_estimator_
        
        import kerastuner as kt

        tuner = kt.Hyperband(
            create_multiple_model_arch,
            objective='val_accuracy',
            max_epochs=3,
            hyperband_iterations=1)
        nlp_data_train, nlp_data_test, meta_data_train, meta_data_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen, multiple=True)
        
        tuner.search((nlp_data_train,meta_data_train),
                     y_train,
                     epochs = params.epochs,
                     validation_data = ((nlp_data_test,meta_data_test), y_test))
        
        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        
        print(f"""The hyperparameter search is complete. 
        The optimal number of units in the first densely-connected layer is {best_hps.get('units')}
        and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.""")
        
        return None, None
    
    elif model_arch == 'word_embedding':
        return None, None
    
    else:
        return None, None

'''# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)
print(test_accuracy)

print(grid_result.best_score_)
print(grid_result.best_params_)
print(grid_result.cv_results_)

results= grid_result
score=grid.score'''
