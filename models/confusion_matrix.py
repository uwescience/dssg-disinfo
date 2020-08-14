from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
import os


def plot_confusion_maatrix(predicted_labels, model_arch):
    ''' Plots confusion matrix based on true vs.
    predicted labels and saves the plot as png in the
    Graph folder. 

    Parameters

    -----------
    obj

        predicted_labels  - csv file containing predicted and validation 
        labels. 

    str
        model - model name such as 'word_embedding', 'basic', 'multiple'

    Returns
    -------
    none
            
    '''

    
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Graphs/')
    
    data = pd.read_csv(predicted_labels)
    
    cm = confusion_matrix(data['label'], data['predicted_label'])
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.title('Confusion Matrix')
    file_name = 'confusion_matrix'+'_'+ model_arch + '.png'
    plt.savefig(results_dir + file_name)

    return 