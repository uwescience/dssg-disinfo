import numpy as np
import pandas as pd

_model_arch_registry = {}

def register_model_arch(arch_name, create_fn, param_names):
    """
    register_model_arch(name, fn, param_names) registers a model architecture
    with the given name so that it can be built with the build_model() function.

    The name should be a string that identifies the model.
    The fn should be a function that accepts the parameters in the param_names and
        yields a sequential model.
    The param_names should be a list of the names of the parameters that fn requires.
    """
    # Save the model
    _model_arch_registry[arch_name] = (create_fn, param_names)

def build_model_arch(arch_name, param_dict):
    """
    Builds a model using the given architecture name and the given parameter
    dictionary.
    """
    # lookup the params and create function:
    (create_fn, params) = _model_arch_registry[arch_name]
    # The f(*[...]) syntax means that instead of being called as f([a,b,c]) the
    # functiion call is converted into f(a, b, c).
    return create_fn(*[param_dict[k] for k in params])
