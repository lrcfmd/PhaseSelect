import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Phase2Vec import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Classification.Model import * #segment_classes_kfold, create_callback, split, class_weights, embedding
from Classification.AtomicModel import Endtoend
from utils import *

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
 
    # Start new calculations

    Tc = 10
    epochs = 100
    dirname = 'magnetic_test'

    # =============== AE RANKING  ================

    # load phase fields of phase vectors if precalculated
    phases = Phase2Vec('', load_phase_vectors='DATA/mpds_magnet_CurieTc.csv', atomic_mode='embed')
    X = np.array([i for i in phases.dt['phases_vectors'].values if len(i)])

    print("SIZE OF ICSD phases: ", phases.dt.shape)

    test = Phase2Vec("", load_phase_vectors='DATA/magnetic_candidates.csv', atomic_mode='embed',maxlength=phases.maxlength)
    print(test.dt.head)
    test_x = np.array([i for i in test.dt['phases_vectors'].values])

    test_x = StandardScaler().fit_transform(test_x)

    def rr(x): return round(x,3)

    #for i in range(300):
    for i in range(1):
        model, history = phases.rank(X, X, epochs=epochs, dirname=f'{dirname}')

        print("Calculating training set rankings...")
        input_x = StandardScaler().fit_transform(X)
        output_x = model.predict(X)
        train_scores = pairwise_distances_no_broadcast(input_x, output_x)
        dtrain = phases.dt[['phases']]
        dtrain = dtrain.assign(rankings=train_scores)
        dtrain.to_csv(f"{dirname}/ranking_all_icsd_single_run.csv")

        print("Predicting test set rankings...")
        pred_scores = model.predict(test_x)
        print("Calculating pairwise distances...")
        rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

        df = test.dt[['phases']]
        df = df.assign(rankings=rankings)

        df.to_csv(f"{dirname}/ranking_magnet_candidates_single_run.csv")

sys.exit(0)
