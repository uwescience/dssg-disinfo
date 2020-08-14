import run_model
from run_model import run_model
from plot_graphs import plot_graphs

if __name__ == "__main__":
    log_file=run_model('basic')
    plot_graphs(log_file, 'basic')