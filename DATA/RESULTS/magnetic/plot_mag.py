import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from matplotlib.colors import Normalize

df = pd.read_csv('mag_ranking_regression_cls.csv')

print(df.head)

fig, ax= plt.subplots()

regres = [int(i) for i in df['predicted Tc']]
markers = [f'${i} K$' for i in regres]
#norm = Normalize(vmin=min(regres), vmax=max(regres))

for i, c in enumerate(regres):
#    color = cm.plasma(norm(c))
    ax.scatter(df['norm. score'].values[i], df['probability'].values[i],
        s=700, color='black',  marker=markers[i])

#fig.colorbar(mappable=None,ax=ax,norm=norm, label='Predicted band gap, eV', cmap=cm.plasma) 
#fig.savefig('/Users/andrij/Desktop/Fig4c_regression.png', dpi=1000)
plt.show()
