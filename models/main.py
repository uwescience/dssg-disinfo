
import run_model
from run_model import run_model

optimizer = ['adam', 'nadam']
bidir_num_filters = [32, 64, 128]
epochs = [3, 5, 7, 9, 11]
dropout_rate = [0.2, 0.3, 0.4, 0.5]

if __name__ == "__main__":
    
    
 #   for opt in optimizer:
  #      for bidir
   #         for

    for (a, b, c, d) in zip(optimizer, bidir_num_filters, epochs, dropout_rate): 
        run_model('multiple', optimizer = a, bidir_num_filters = b, epochs = c, dropout_rate = d)
        

    log_file=run_model('multiple', optimizer = a, bidir_num_filters = b, epochs = c, dropout_rate = d)
    plot_graphs(log_file, 'multiple')