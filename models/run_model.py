import model_arch
from model_arch import *
import baseline_model
from baseline_model import *
import build_model
from build_model import *
import param_tune
from param_tune import *
from get_data import *
from word_embedding_arch import *

# Importing the default parameters
import params_class
params=params_class.params()

def run_model(model_arch='basic', **copacabana):
    """Run a model type specified by the model_arch.
    default parameters stored in the params_class
    will be used if not overwritten by user in **copacabana
    
    input
    -----
    model_arch: string, the type of the model
    **copacabana: parameteres
    
    output
    ------
    history: model history, includes train and validation accuracy, train and validation loss
    fitted_model: final model
    """
    
    # Calling the default parameters
    default_params = {value:items for value, items in params.__dict__.items()}
    # Storing all the parameters into copacabana, parameters passed by user will overwrite default
    copacabana = {k: copacabana.get(k, default_params[k]) for k in default_params.keys()}
    
    if model_arch == 'basic': # basic model- LSTM
        
        file_name=param_tune(model_arch, **copacabana) #returns file name with epoch logs for the best model
        plot_graphs(file_name)

    elif model_arch == 'multiple': # two input model- linguistic features and text input
        
        model=build_model(model_arch=model_arch, **copacabana)
        compiled_model=compile_model(model)
        history, fitted_model = fit_and_run_model(compiled_model, vocab_size=10000, maxlen=681, epochs=10, model_arch=model_arch)
    
    elif model_arch == 'word_embedding': # word embedding model, pulls in word_embedding file specified by user.
        history, fitted_model=param_tune(model_arch)
        
    else:
        print("Invalid model type entered entered!")
        history=None
        fitted_model=None
    
    return history, fitted_model
