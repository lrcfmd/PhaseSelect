import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from matplotlib.colors import Normalize


df = pd.read_csv('_bg_P075_R01_regression.csv')

df =df[df['norm. score'] < 0.095]
fig, ax= plt.subplots()
#plt.scatter(df['norm. score'], df['av. probability'], s=df['error bar'].values*1000, c=df['regression'].values, cmap='plasma')
regres = [round(i,1) for i in df['regression']]
markers = [f'${i} eV$' for i in regres]
norm = Normalize(vmin=min(regres), vmax=max(regres))

for i, c in enumerate(regres):
    color = cm.plasma(norm(c))
    ax.scatter(df['norm. score'].values[i], df['av. probability'].values[i],
        s=700, color='black',  marker=markers[i])

#fig.colorbar(mappable=None,ax=ax,norm=norm, label='Predicted band gap, eV', cmap=cm.plasma) 

#fig.savefig('/Users/andrij/Desktop/Fig4c_regression.png', dpi=1000)
plt.show()
