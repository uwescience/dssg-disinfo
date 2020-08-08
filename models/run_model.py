import model_arch
from model_arch import register_model_arch, build_model_arch

import baseline_model
from baseline_model import *

#import get_data
#from get_data import get_data_and_split()

import build_model
from build_model import build_model, compile_model, fit_and_run_model

#(X_train, X_test, y_train, y_test) = get_data_and_split(vocab_size=10000, maxlen=681)

def run_model(model_arch='basic'):
    model= build_model(model_arch=model_arch)
    compiled_model= compile_model(model)
    fitted_model= fit_and_run_model(compiled_model)
    return compiled_model, fitted_model
