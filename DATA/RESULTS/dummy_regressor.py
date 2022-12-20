import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score

def plot_dummy(d, true, r2, name):
    errors = np.abs([dd - tt for dd,tt in zip(d, true)])
    plt.scatter(true, d, c=errors, label=fr'Dummy r$^2$ = {r2}', cmap='inferno')
    plt.colorbar(label='Absolute error, |K|')
    plt.xlabel(f'True Values [{name}, K]',fontsize=12)
    plt.ylabel(f'Predictions [{name}, K]',fontsize=12)
    lims = [min(true), max(true)]
    _ = plt.plot(lims, lims, 'black', label=r'r$^2$ = 1')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend()
    plt.show()


def dummy(df, prop='Tc'):
    predict = df[f'Predicted {prop}'].values
    true = df[f'True {prop}'].values
    m = np.mean(true)
    mm = np.array([m for i in range(len(predict))])
    mae =  MAE(mm, true)
    r2 = r2_score(mm, true)
    plot_dummy(mm, true, r2, prop) 
    return mae, mae/(max(true) - min(true))

df = pd.read_csv(sys.argv[1])
#m, mr = dummy(df, 'Curie T')
m, mr = dummy(df, 'Tc')
print('MAE:', m)
print('MAE/Range:', mr)


