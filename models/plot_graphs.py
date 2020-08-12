import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_graphs(history, string):
    ''' Plots model output metrics such as 
    accuracy, etc. for a given model and
    saves a png file with the results. 
    The name of the file will be a string =
    history+metric

    Parameters

    -----------
    obj
        history object collected after an instance of 
        a model has been fit 
    str
        string passed as the name of the model (i.e. accuracy, loss)

    Returns
    -------
    none
            
    
    '''
    
    #history=collected model, string= 'accuracy' or whichever. 
    #It will name the png output file as history+metric
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    file_name= f'{history}'+ string + '.png'
    plt.savefig(file_name)
    return
