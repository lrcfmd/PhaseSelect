# Classify icsd phases 
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Phase2Vec import Phase2Vec
from DB.parse_phases import parse_phases
from DB.augmentation import augment
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Classification.Model import * #segment_classes_kfold, create_callback, split, class_weights, embedding
from Classification.AtomicModel import Endtoend
from PostProcess.plotting_results import *
from stat_models import pairwise_distances_no_broadcast

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
 
    Tc = 4.5
    epochs = 100
    #dirname = 'BANDGAP/Reproduce'
    dirname = 'BANDGAP/Attention'
    log = 'bg_class_log'
#   # Load phase vectors if precalculated
    phases = Phase2Vec('', load_phase_vectors='DB/BandGap/mpds_phases_band_gap.csvone_hot_phases.pkl')
    print('size of onehot:', phases.dt['onehot'].values[0].shape)

#    print(test.dt.head())
 
    # Training set - all phase fields (vectorized)
    # training = phases.dt['phases'].values
    X = np.array([i for i in phases.dt['onehot'].values])
    y = np.where(phases.dt['max energy gap'].values > Tc, 1, 0)

    # Test
    #test = Phase2Vec("", load_phase_vectors='Unexplored_ternaries.pkl', maxlength=10)


    for i in range(1): 
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

        sys.exit() 
        # =============== TEST ============================= 
        print('size of onehot:', test.dt['onehot'].values[0].shape)
        print("Predicting classification for unexplored phase fields...")
        test_x = np.array([i for i in test.dt['onehot'].values])
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability
     
        dttest = test.dt.assign(probability_Tc300K=y_test)
        dttest = dttest[['phases','probability_Tc300K']]
        dttest = dttest.sort_values(by='probability_Tc300K', ascending=False)
        dttest.to_csv(f'{dirname}/unexplored_ternaries_classification_run{i}.csv', index=None)
