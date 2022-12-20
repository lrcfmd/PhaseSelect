import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from matplotlib.colors import Normalize



f1 ='MPDS_magnetic_phases.csv'
f2 ='mpds_phases_band_gap.csv'
f3 ='Supercon_phases.csv'

def plot_n(f, name, n, plot=True):
    df = pd.read_csv(f)
    if plot:
        plt.hist(df['n compositions'].values, n, label=f'{name}')
    else:
        vals, edges = np.histogram(df['n compositions'].values, n)
        return vals


vals_band = plot_n(f2, 'Band gap', 100, False)

vals_mag = plot_n(f1, 'Magnetic', 1000, True)
vals_scon =plot_n(f3, 'Supercon', 1000, True)


plt.bar(x=np.arange(1,101,1), height=vals_band/20, width=0.1, align='edge', label=r'1/20 Band gap', color='tab:green')

plt.xlim([0,10])
plt.xlabel('N compositions in a phase field')
plt.ylabel('N phase fields')
plt.legend()
plt.show()
