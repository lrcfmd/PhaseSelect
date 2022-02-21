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
 
    Tc = 10
    epochs = 100
    dirname = 'SUPERCON/Attention'
    log = 'scon_class_log'
#   # Load phase vectors if precalculated
    phases = Phase2Vec('', load_phase_vectors='DB/DATA/combined_mpds_scon_Tc.pklone_hot_phases.pkl')
    print('size of onehot:', phases.dt['onehot'].values[0].shape)
 
    # Training DATA set - all phase fields (vectorized)
    X = np.array([i for i in phases.dt['onehot'].values])
    y = np.where(phases.dt['max Tc'].values > Tc, 1, 0)

    # Testing
    test = Phase2Vec("", load_phase_vectors='Unexplored_ternaries.pklone_hot_phases.pkl', maxlength=8)
    print('size of onehot:', test.dt['onehot'].values[0].shape)
    test_x = np.array([i for i in test.dt['onehot'].values])

    for i in range(1):
        # =============== MODEL CLASSIFY  ================
        model, model_att = phases.classifier(X, y, epochs=epochs, dirname=f'{dirname}/weights/{i}')
        print("Getting attention from training phase fields...")
        y_att = model_att.predict(X)

        # sum over heads
        #sum_att = y_att.sum(axis=1)
        #attention_scores = [sum_att[i,:,:] for i in range(X.shape[0])]

        # use single 0th head:
        attention_scores = [y_att[i,0,:,:] for i in range(X.shape[0])]

        att = phases.dt.assign(attention=attention_scores)
        att = att[['phases','max Tc', 'attention']]
        att.to_pickle(f'{dirname}/supercon_phases_Tc10_attention.pkl')

        # =============== TEST =============================
        print('size of onehot:', test.dt['onehot'].values[0].shape)
        print("Predicting classification for unexplored phase fields...")
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability

        #print("Predicting attention for unexplored phase fields...")
        #y_att = model_att.predict(test_x)
        #attention_scores = [np.array(y_att[i,0,:,:])+np.array(y_att[i,1,:,:]) for i in range(test_x.shape[0])]

        dttest = test.dt.assign(probability_Tc10K=y_test)
        #dttest = dtest.assign(attention=attention_scores)
        dttest = dttest[['phases','probability_Tc10K']]
        dttest.to_csv(f'{dirname}/unexplored_ternaries_classification_run{i}.csv', index=None)
