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

def rr(x): return round(x,3)

if __name__ == '__main__':

    # =============== AE RANKING  ================
    epochs = 100
    dirname = 'test'

    # =============== Training set
    phases = Phase2Vec('', load_phase_vectors='DATA/mpds_magnet_CurieTc.csv', atomic_mode='embed')
    X = np.array([i for i in phases.dt['phases_vectors'].values if len(i)])

    # =============== Testing set
    test = Phase2Vec("", load_phase_vectors='DATA/magnetic_candidates.csv', atomic_mode='embed',maxlength=phases.maxlength)
    test_x = np.array([i for i in test.dt['phases_vectors'].values])
    test_x = StandardScaler().fit_transform(test_x)

    # Choose a size of an ensemble of models
    ensemble_n = 1
    for i in range(ensemble_n):
        model, history = phases.rank(X, X, epochs=epochs, dirname=f'{dirname}')

        print("Calculating training set rankings...")
        input_x = StandardScaler().fit_transform(X)
        output_x = model.predict(X)
        train_scores = pairwise_distances_no_broadcast(input_x, output_x)
        dtrain = phases.dt[['phases']]
        dtrain = dtrain.assign(rankings=train_scores)
        dtrain.to_csv(f"{dirname}/ranking_mpds_magnet_single_run.csv")

        print("Predicting test set rankings...")
        pred_scores = model.predict(test_x)
        print("Calculating pairwise distances...")
        rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

        df = test.dt[['phases']]
        df = df.assign(rankings=rankings)

        df.to_csv(f"{dirname}/ranking_magnet_candidates_single_run.csv")
