import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score


def pl(data, name='Band gap eV',color_name='viridis'):
    y_test = data[f'True {name}']
    y_predict = data[f'Predicted {name}'] 
    errors = [abs(p-t) for p,t in zip(y_predict, y_test)]

    # r2
    r2 = round(r2_score(y_test, y_predict),2)
    print(r'r$^2$ =', r2)


    # plot
    #plt.scatter(y_test, y_predict, s=errors, c=errors, label=fr'r$^2$ = {r2}')
    plt.scatter(y_test, y_predict, c=errors, label=fr'r$^2$ = {r2}', cmap=color_name)
    plt.colorbar(label='Absolute error, |K|')
    #plt.colorbar(label='Absolute error, |eV|')
    plt.xlabel(f'True Values [{name}, K]',fontsize=12)
    plt.ylabel(f'Predictions [{name}, K]',fontsize=12)
    lims = [min(y_test), max(y_test)]
    plt.xlim(lims)
    plt.ylim(lims)
    #plt.ylim([i/2 for i in lims])
    _ = plt.plot(lims, lims, 'black', label=r'r$^2$ = 1')

    plt.legend(fontsize=12)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    pl(df, 'Curie T',color_name='winter')
    #pl(df, 'Tc, K',color_name='autumn')
    #pl(df, 'Band gap',color_name='summer')
    #pl(df)
