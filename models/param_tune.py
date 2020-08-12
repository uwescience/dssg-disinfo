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

#import get_data
#from get_data import get_data_and_split

#import run_model
#from run_model import run_model
import baseline_model
from baseline_model import create_basic_model_arch

def param_tune(model):
    '''Runs parameter tuning..'''
    import get_data
    from get_data import get_data_and_split
    #Parameter grid for grid search
    #params = {k:v for (k,v) in copacabana.items() if k in **copacabana}
    
    param_grid = dict(bidir_num_filters=[32, 64, 128],
                  dense_1_filters=[10],
                    vocab_size=[10000],
                      embedding_dim=[300],
                      maxlen=[681],
                  optimizer=['adam','nadam']) 
    
    model_new = KerasClassifier(build_fn=create_basic_model_arch)
    
    grid = RandomizedSearchCV(estimator=model_new, param_distributions=param_grid, 
                        cv = StratifiedKFold(n_splits=5), verbose=1, n_iter=5, scoring='accuracy')
    #pull in data
    X_train, X_test, y_train, y_test = get_data_and_split(params.vocab_size, params.maxlen)
    #model_new.fit(X_train, y_train)
    grid_result = grid.fit(X_train, y_train)
    test_accuracy = grid.score(X_test, y_test)
    
    return grid.best_estimator_


'''# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)
print(test_accuracy)

print(grid_result.best_score_)
print(grid_result.best_params_)
print(grid_result.cv_results_)

results= grid_result
score=grid.score'''
