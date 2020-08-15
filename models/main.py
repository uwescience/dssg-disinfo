import run_model
from run_model import run_model
from plot_graphs import plot_graphs

epochs = [3,5,7, 9, 11]
if __name__ == "__main__":
for epoch in epochs:        
    log_file = run_model('basic', epochs=epoch)
    plot_graphs(log_file, 'basic')