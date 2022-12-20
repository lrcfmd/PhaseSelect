import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from Phase2Vec_simple import Phase2Vec
from utils import *
import matplotlib.pyplot as plt
from utils_calibration import *
from calibrator import *

# dataset and model parameters
prop = 'max Tc' 
Tc = 10
epochs = 400
dirname = 'calibration'
filename = 'DATA/combined_mpds_scon_Tc.pkl'
# Choose a size of an ensemble of models
ensemble_n = 2

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
    # Training set - all phase fields
    if os.path.exists(f'{filename}one_hot_phases.pkl'):
        phases = Phase2Vec('', load_phase_vectors=f'{filename}one_hot_phases.pkl')
    else:
        phases = Phase2Vec('', load_phase_vectors=f'{filename}')

    X0 = np.array([i for i in phases.dt['onehot'].values])
    y0 = np.where(phases.dt[f'{prop}'].values > Tc, 1, 0)

    calibrators = [HistogramCalibrator(n_bins=10),
                   PlattCalibrator(log_odds=True),
                   PlattHistogramCalibrator(n_bins=10, log_odds=True)]

    for i in range(ensemble_n):

        _, names_test, _, _ = train_test_split(phases.dt['phases'], y0, test_size=0.2) #random_state=i)
        X, X_test, y, y_test = train_test_split(X0, y0, test_size=0.2) #random_state=i)

        # =============== MODEL CLASSIFY  ================
        model, represent = phases.classifier(X, y, epochs=epochs)

        # =============== TEST =============================
        y_predict = model.predict(X_test)
        y_predict = y_predict[:,1] # look at the '1' class probability
        
        # =============== CALIBRATE ========================
        print(f'ensemble: {i} Calibration summary')
        tmp_df = pd.DataFrame({'phases': names_test, 'y_true': y_test, 'y_pred': y_predict})
        tmp_df.to_pickle('scon_calibration_raw.pkl')
        tmp_df = {'scon class':tmp_df} 
        calib_res = compute_calibration_summary(tmp_df, 'y_true', 'y_pred', n_bins=10)
        print(calib_res.head())
        n_bins = 10

        if i == 0:
            for cal in calibrators:
                cal.fit(y_predict, y_test)
        else:
            for cal, name in zip(calibrators,['hist', 'platt', 'plathist']):
                y_pred = cal.predict(y_predict)
                print(f'ensemble: {i} Post-calibration summary')
                tmp_df = pd.DataFrame({'phases': names_test, 'y_true': y_test, 'y_pred': y_predict})
                tmp_df.to_pickle(f'scon_calibration_{name}.pkl')
                tmp_df = {f'{name}':tmp_df}
                calib_res = compute_calibration_summary(tmp_df, 'y_true', 'y_pred', n_bins=10)
                print(calib_res.head())

