import model_arch
from model_arch import register_model_arch, build_model_arch

import baseline_model
from baseline_model import *

import m_m
from m_m import *

#import get_data
#from get_data import get_data_and_split()

import build_model
from build_model import *

#import param_tune
#from param_tune import param_tune

import params_class
params=params_class.params()

def run_model(model_arch='basic', **copacabana):
    
    default_params = {value:items for value, items in params.__dict__.items()}
    copacabana = {k: copacabana.get(k, default_params[k]) for k in default_params.keys()}
    
    # Ask user if they need hypertuning
    #hypertuning_choice=input("Do you want hypertuning?y/n:")
    
    if model_arch == 'basic':
        
        model= build_model(model_arch=model_arch, **copacabana)
        compiled_model= compile_model(model)
        if hypertuning_choice == 'y':
            hypertuned_compiled_model=param_tune(compiled_model)
            # Need to pull in the outputs from tuning as inputs here:
            #history, fitted_model= fit_and_run_model(hypertuned_compiled_model)
        else:
            history, fitted_model= fit_and_run_model(compiled_model)
    
    elif model_arch == 'multiple':
        #model=build_model(model_arch=model_arch, **copacabana)
        #compiled_model=compile_model(model)
        file_name = fit_and_run_model(create_multiple_model_arch(**copacabana), model_arch=model_arch, **copacabana)
        
    
    elif model_arch == 'word_embedding':
        embedding_path=input("Enter path of word embedding:")
        history, fitted_model=fit_and_run_embedding_model(embedding_path=embedding_path, embedding_dim=300, maxlen=681, epochs=10, model_arch=model_arch)
        
    else:
        print("Invalid model type entered entered!")
        history=None
    
    return file_name
