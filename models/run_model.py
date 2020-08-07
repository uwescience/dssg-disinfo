import baseline
from baseline import *

def run_model(model_arch='basic'):
    model= build_model(model_arch=model_arch)
    compiled_model= compile_model(model)
    fitted_model= fit_and_run_model(compiled_model)
    return model, compiled_model, fitted_model
