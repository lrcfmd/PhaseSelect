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
 
    # Start new calculations

    # Specify threshold value
    Tc = 300
    epochs = 100

    # Specify data files
    dirname = 'generic'
    log = 'generic_class_log'
    training_data = 'DATA/mpds_magnet_CurieTc.csv'
    test_data = 'DATA/Quaternary_oxides.csv'

    os.makedirs(dirname, exist_ok=True)

    # Load phase vectors
    phases = Phase2Vec('', load_phase_vectors=training_data)
    test = Phase2Vec("", load_phase_vectors=test_data, maxlength=phases.maxlength)
 
    # Training set - all phase fields (vectorized)
    X = np.array([i for i in phases.dt['onehot'].values])
    y = np.where(phases.dt['max Tc'].values > Tc, 1, 0)

    # Choose a size of ensemble of models to average the results over
    ensemble_n = 1
    for i in range(ensemble_n): 
        # =============== MODEL CLASSIFY  ================
        os.makedirs(f'{dirname}/weights',exist_ok=True)
        model, model_att = phases.classifier(X, y, epochs=epochs, dirname=f'{dirname}/weights/{i}')
        print("Getting attention from training phase fields...")
        y_att = model_att.predict(X)

        # sum over heads
        sum_att = y_att.sum(axis=1)
        attention_scores = [sum_att[i,:,:] for i in range(X.shape[0])]

        att = phases.dt.assign(attention=attention_scores)
        att = att[['phases', 'max Tc', 'attention']]
        att.to_pickle(f'{dirname}/phases_attention.pkl')

        # =============== TEST ============================= 
        print('size of onehot:', test.dt['onehot'].values[0].shape)
        print("Predicting classification for unexplored phase fields...")
        test_x = np.array([i for i in test.dt['onehot'].values])
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability
     
        dttest = test.dt.assign(probability_Tc=y_test)
        dttest = dttest[['phases','probability_Tc']]
        dttest = dttest.sort_values(by='probability_Tc', ascending=False)
        dttest.to_csv(f'{dirname}/classification_results_run_{i}.csv', index=None)
