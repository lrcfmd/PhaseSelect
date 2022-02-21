import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Phase2Vec import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Classification.Model import 
from Classification.AtomicModel import Endtoend
from utils import *

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
 
    # Start new calculations

    Tc = 300
    epochs = 100
    dirname = 'test'
    log = 'mag_class_log'
#   # Load phase vectors if precalculated
    phases = Phase2Vec('', load_phase_vectors='DATA/mpds_magnet_CurieTc.csvone_hot_phases.pkl')
    print('size of onehot:', phases.dt['onehot'].values[0].shape)

    test = Phase2Vec("", load_phase_vectors='DATA/Ternary_phase_fields.pkl',maxlength=phases.maxlength)
 
    # Training set - all phase fields (vectorized)
    X = np.array([i for i in phases.dt['onehot'].values])
    y = np.where(phases.dt['max Tc'].values > Tc, 1, 0)

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
        att = att[['phases','max Tc', 'attention']]
        att.to_pickle(f'{dirname}/magnet_phases_attention.pkl')

        # =============== TEST ============================= 
        print('size of onehot:', test.dt['onehot'].values[0].shape)
        print("Predicting classification for unexplored phase fields...")
        test_x = np.array([i for i in test.dt['onehot'].values])
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability
     
        dttest = test.dt.assign(probability_Tc300K=y_test)
        dttest = dttest[['phases','probability_Tc300K']]
        dttest = dttest.sort_values(by='probability_Tc300K', ascending=False)
        dttest.to_csv(f'{dirname}/run_{i}.csv', index=None)
