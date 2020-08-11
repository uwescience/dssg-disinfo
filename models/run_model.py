import model_arch
from model_arch import register_model_arch, build_model_arch

import baseline_model
from baseline_model import *

#import get_data
#from get_data import get_data_and_split()

import build_model
from build_model import *

#(X_train, X_test, y_train, y_test) = get_data_and_split(vocab_size=10000, maxlen=681)

def run_model(model_arch='basic', **copacabana):
    # Ask user if they need hypertuning
    hypertuning_choice=input("Do you want hypertuning?y/n.")
    
    if model_arch == 'basic':
        model= build_model(model_arch=model_arch, **copacabana)
        compiled_model= compile_model(model)
        if do_you_want_hypertuning == 'y':
            hypertuned_compiled_model=I_will_hypertune_you(compiled_model)
            history, fitted_model= fit_and_run_model(hypertuned_compiled_model)
        else:
            history, fitted_model= fit_and_run_model(compiled_model)
    
    elif model_arch == 'multiple':
        model=build_model(model_arch=model_arch, **copacabana)
        compiled_model=compile_model(model)
        history, fitted_model = fit_and_run_model(compiled_model, vocab_size=10000, maxlen=681, epochs=10, model_arch=model_arch)
    
    elif model_arch == 'word_embedding':
        embedding_path=input("Enter path of word embedding:")
        history, fitted_model=fit_and_run_embedding_model(embedding_path=embedding_path, embedding_dim=300, maxlen=681, epochs=10, model_arch=model_arch)
        
    else:
        print("Invalid model type entered entered!")
    
    return history
