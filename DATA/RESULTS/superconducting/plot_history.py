import pickle
import sys
import matplotlib.pyplot as plt

def plot_history( history, prop='loss'):
        plt.plot(history[f'{prop}'], label='train')
        plt.plot(history[f'val_{prop}'], label='test')
        plt.xlabel('Epoch')
        #plt.ylabel(f'MAE [Band gap, eV]')
        #plt.ylabel(f'MAE [Curie T, K]')
        plt.ylabel(r'MAE [Transition T$_c$, K]')
        #plt.ylim([0.2,0.7])
        plt.legend()
        plt.show()

with open(sys.argv[1], 'rb') as history_file:
    history = pickle.load(history_file)

plot_history(history)
