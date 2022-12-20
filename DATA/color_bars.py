import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from statistics import stdev

fig, ax = plt.subplots()

def spread(f, prop='max Tc'):
    """ spread of values """
    try:
        df = pd.read_pickle(f)
    except:
        df = pd.read_csv(f)
    return df[f'{prop}'].max() - df[f'{prop}'].min()


def gradientbars(bars, errors):
    # set up the gradient for the cmap
    grad = np.atleast_2d(np.linspace(0,1,256)).T
    # reestablish the plot area
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    #ax.axis(lim)
    # color each bar
    cmaps = ['winter','autumn','summer']
    cmaps = [cm.get_cmap(i).reversed() for i in cmaps]
    labs = ['Magnetic','Superconducting', 'Band gap']
    for bar,c,l,yerr in zip(bars,cmaps+cmaps, labs+labs, errors):
        bar.set_facecolor("none")
        x,y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", cmap=c, label=l, zorder=2)
        ax.errorbar(x+w/2, y+h, yerr=yerr, linewidth=1, capsize=2, zorder=3, ecolor=c(0.5))


# DATA
mag_rf = [163.28452154152066, 165.09349999269247, 162.87150020831885, 174.32819220325908, 167.0416937021449]
scon_rf = [12.308787884740012, 9.881140006197908, 8.93487242495497, 9.080692692786322, 9.73600162624044]
bg_rf = [0.55, 0.51, 0.5, 0.48, 0.51]
#
mag_nn = [120.01, 119.4321, 125.12328, 133.0198, 137.9862]
scon_nn = [10.40710302591186, 8.186316676344429, 7.4052409070914065, 6.978677283882736, 7.1903784946911085] 
bg_nn = [0.42775382192730903625,0.4441090524196625,0.44257652759552,0.448235422372818,0.4414575397968292]
#
mag=[np.average(mag_nn), np.average(mag_rf)]
scon=[np.average(scon_nn), np.average(scon_rf)]
bg=[np.average(bg_nn), np.average(bg_rf)]
#
varmag = spread('mpds_magnet_CurieTc.csv')
varscon = spread('Supercon_phases.csv')
varbg = spread('mpds_phases_band_gap.csv', 'max energy gap')
#
mag =  np.array(mag) / varmag
scon = np.array(scon) / varscon
bg =   np.array(bg) / varbg
#
mag_rf = np.array(mag_rf) / varmag
mag_nn = np.array(mag_nn) / varmag
print('MAG', mag_nn, np.average(mag_nn))
scon_rf = np.array(scon_rf) / varscon
scon_nn = np.array(scon_nn) / varscon
print('Scon', scon_nn, np.average(scon_nn))
bg_rf = np.array(bg_rf) / varbg
bg_nn = np.array(bg_nn) / varbg
print('bg', bg_nn, np.average(bg_nn))
#
errors = [stdev(mag_rf), stdev(scon_rf), stdev(bg_rf), stdev(mag_nn), stdev(scon_nn), stdev(bg_nn)]

# DUMMY
d_mag = 0.14741302004701123
d_scon = 0.13742662940662
d_bg = 0.10272607871884773
d_df = pd.DataFrame({'a': [-3, -2, -1], 'b':[d_mag, d_scon, d_bg]})
gradientbars(ax.bar(d_df.a, d_df.b, edgecolor='grey'), [0,0,0])
plt.xticks([-2, 2, 6], ['Dummy', 'Random Forest', 'PhaseSelect'], fontsize=12)
plt.yticks([.02, .04, .06, .08, 0.1, .12, .14], [2,4,6,8,10, 12, 14])
#=============

# PLOTTING
df = pd.DataFrame({'a':[1,2,3,5,6,7], 'b':[mag[1],scon[1],bg[1],mag[0],scon[0],bg[0]]})
ax.grid(axis='y', zorder=0)
gradientbars(ax.bar(df.a, df.b, edgecolor='grey'), errors)

#plt.xticks([2, 6], ['Random Forest', 'PhaseSelect'], fontsize=12)
#plt.yticks([.02, .04, .06, .08], [2,4,6,8])
#plt.yticks([.02, .04, .06, .08])
plt.ylabel(r'100% $\times$ MAE / Values Range', fontsize=14)
plt.xlim([-3.5, 7.5])
plt.ylim([0, 0.16])
plt.show()
