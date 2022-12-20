import sys
import numpy as np
np.random.seed(1)
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Phase2Vec_simple import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from utils import *
import matplotlib.pyplot as plt

dirname = 'validation'
#training_data = 'DATA/Supercon_phases.csvone_hot_phases.pkl'        # SCON
#training_data = 'DATA/combined_mpds_scon_Tc.pklone_hot_phases.pkl'  # SCON
#training_data = 'DATA/mpds_magnet_CurieTc.csvone_hot_phases.pkl'    # MAG
training_data = 'DATA/mpds_phases_band_gap.csv'                      # BG


if __name__ == '__main__':
    # Check files exist:
    for f in [dirname, training_data]:
        if not os.path.exists(f'{f}'):
            print(f'{f}: path does not exist. Exiting. ')
            sys.exit(0)

    # RF VS NN ==============================
    # RF
    phases = Phase2Vec('', load_phase_vectors=training_data, dirname=dirname, attention=False, atomic_mode='magpie') 
    X0 = np.array([i for i in phases.dt['phases_vectors'].values])
    
    # NN
    # phases = Phase2Vec('', load_phase_vectors=training_data, dirname=dirname, attention=False, atomic_mode='envs')   
    # X0 = np.array([i for i in phases.dt['onehot'].values])
    
    # K-fold validate
    #  ==============================
    #y0 = phases.dt['max Tc'].values                               # SCON or MAG
    y0 = phases.dt['max energy gap'].values                        # BG

    k_errors = phases.validate(phases.rf_regressor, X0, y0)
    #k_errors = phases.validate(phases.regressor, X0, y0, epochs=2000, k_features=140, batch_size=40, verbose=1)
 
    print(k_errors, np.average(k_errors))
