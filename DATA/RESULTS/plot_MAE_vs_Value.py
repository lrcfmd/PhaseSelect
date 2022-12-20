import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score

def pl(data, name='Band gap, eV'):
    df = pd.read_csv(data)
    y_test = data['true Tc']
    #y_test = np.expm1(y_test)
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
    lims = [0, 1500]
    lims = [0, 160]
    #lims = [1, 7]
    plt.xlim(lims)
    plt.ylim(lims)
    #plt.ylim([i/2 for i in lims])
    _ = plt.plot(lims, lims, 'black', label=r'r$^2$ = 1')

if __name__ == "__main__":
    d1 =  'bg_regressor_predictions_r2.csv'
    d2 =  ''
    d3 =  ''
    pl(d1, 'Tc, K')
    #pl(d1, 'Tc, K')
    #pl(d1, 'Tc, K')
    plt.xlabel(f'')
    plt.ylabel(f'Predictions [{name}]')
    plt.legend()
    plt.show()
