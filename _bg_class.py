import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Phase2Vec import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Classification.Model import *
from Classification.AtomicModel import Endtoend
from utils import *

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
 
    Tc = 4.5
    epochs = 100
    dirname = 'test'
    log = 'bg_class_log'
   
 
    # Training set - all phase fields
    # Load one-hot representations if precalculated
    if os.path.exists('DATA/mpds_band_gap.csvone_hot_phases.pkl'):
        phases = Phase2Vec('', load_phase_vectors='DATA/mpds_band_gap.csvone_hot_phases.pkl')
    else:
        phases = Phase2Vec('', load_phase_vectors='DATA/mpds_band_gap.csv')
    X = np.array([i for i in phases.dt['onehot'].values])
    y = np.where(phases.dt['max energy gap'].values > Tc, 1, 0)

    # Test
    if os.path.exists('DATA/Ternary_phase_fields.pklone_hot_phases.pkl'):
        test = Phase2Vec("", load_phase_vectors='DATA/Ternary_phase_fields.pklone_hot_phases.pkl',maxlength=phases.maxlength)
    else:
        test = Phase2Vec("", load_phase_vectors='DATA/Ternary_phase_fields.pkl',maxlength=phases.maxlength)

    # Choose a size of an ensemble of models
    ensemble_n = 1
    for i in range(ensemble_n): 
        # =============== MODEL CLASSIFY  ================
        model, model_att = phases.classifier(X, y, epochs=epochs, dirname=f'{dirname}/weights/{i}')
        print("Getting attention from training phase fields...")
        y_att = model_att.predict(X)

        # sum over heads
        sum_att = y_att.sum(axis=1)
        attention_scores = [sum_att[i,:,:] for i in range(X.shape[0])]

        att = phases.dt.assign(attention=attention_scores)
        att = att[['phases','max energy gap', 'attention']]
        att.to_pickle(f'{dirname}/bg_phases_attention.pkl')

        # =============== TEST ============================= 
        print('size of onehot:', test.dt['onehot'].values[0].shape)
        print("Predicting classification for unexplored phase fields...")
        test_x = np.array([i for i in test.dt['onehot'].values])
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability
     
        dttest = test.dt.assign(probability_large_bg=y_test)
        dttest = dttest[['phases','probability_large_bg']]
        dttest = dttest.sort_values(by='probability_large_bg', ascending=False)
        dttest.to_csv(f'{dirname}/unexplored_ternaries_bg_classification_run{i}.csv', index=None)
