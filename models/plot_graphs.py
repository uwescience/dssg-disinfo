<<<<<<< HEAD
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os




def plot_graphs(log_file, model):
=======
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_graphs(history, string):
>>>>>>> 511efc3bca97b594df31ab06eb19d1bc5454bba5
    ''' Plots model output metrics such as 
    accuracy, etc. for a given model and
    saves a png file with the results. 
    The name of the file will be a string =
    history+metric

    Parameters

    -----------
    obj
<<<<<<< HEAD
        log_file  - csv log containing validation accuracy, 
        loss, etc. collected after an instance of a model has been fit 
    str
        model - model name (i.e. 'word_embedding', 'basic', 'multiple')
=======
        history object collected after an instance of 
        a model has been fit 
    str
        string passed as the name of the model (i.e. accuracy, loss)
>>>>>>> 511efc3bca97b594df31ab06eb19d1bc5454bba5

    Returns
    -------
    none
            
    
    '''
<<<<<<< HEAD
    script_dir = os.path.dirname(__file__)

    results_dir = os.path.join(script_dir, 'Graphs/')
    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    data = pd.read_csv(log_file, sep=';')
    
  
    
    # plotting loss 
    loss = plt.figure(figsize=(15,8))
    plt.plot(data['epoch'], data[['val_loss','loss']])
    plt.legend(data[['val_loss','loss']])
    plt.title('Loss for Training and Validation' + ' - ' + model)
    plt.xlabel("Epochs")
    plt.ylabel("Loss") 
    file_name_loss= model + '_'+ 'loss' + '.png'
    plt.savefig(results_dir + file_name_loss)
   

        # plotting accuracy 
    plt.figure(figsize=(15,8))
    accuracy= plt.plot(data['epoch'], data[['val_accuracy','accuracy']])
    plt.legend(data[['val_accuracy','accuracy']])
    plt.title('Accuracy for Training and Validation' + ' - ' + model)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")   
    file_name_acc= model + '_'+ 'accuracy' + '.png'
    plt.savefig(results_dir + file_name_acc)
       
    
    
=======
    
    #history=collected model, string= 'accuracy' or whichever. 
    #It will name the png output file as history+metric
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    file_name= f'{history}'+ string + '.png'
    plt.savefig(file_name)
>>>>>>> 511efc3bca97b594df31ab06eb19d1bc5454bba5
    return
