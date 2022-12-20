import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score


def pl(data, name='Band gap, eV'):
    y_test = data['true Tc']
    y_predict = data['predicted Tc'] 
    errors = [abs(p-t) for p,t in zip(y_predict, y_test)]

    # r2
    r2 = round(r2_score(y_test, y_predict),2)
    print(r'r$^2$ =', r2)

    # plot
    #plt.scatter(y_test, y_predict, s=errors, c=errors, label=fr'r$^2$ = {r2}')
    plt.scatter(y_test, y_predict, c=errors, label=fr'r$^2$ = {r2}')
    plt.colorbar(label='Absolute error, |K|')
    #plt.colorbar(label='Absolute error, |eV|')
    plt.xlabel(f'True Values [{name}]')
    plt.ylabel(f'Predictions [{name}]')
    lims = [min(y_test), max(y_test)]
    plt.xlim(lims)
    plt.ylim(lims)
    #plt.ylim([i/2 for i in lims])
    _ = plt.plot(lims, lims, 'black', label=r'r$^2$ = 1')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    pl(df, 'Tc, K')
