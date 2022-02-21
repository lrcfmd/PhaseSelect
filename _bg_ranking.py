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
 
    # Start new calculations
    #phases = Phase2Vec('DB/Supercon_data.csv')

    Tc = 10
    epochs = 100
    dirname = 'BANDGAP/Reproduce_ranking'

    # =============== AE RANKING  ================

    # load phase fields of phase vectors if precalculated
    phases = Phase2Vec('', load_phase_vectors='DB/BandGap/mpds_phases_band_gap.csv',atomic_mode='embed')
    X = np.array([i for i in phases.dt['phases_vectors'].values if len(i)])

    print(phases.dt.head)
    test = Phase2Vec("", load_phase_vectors='Unexplored_ternaries.pkl', maxlength=200, atomic_mode='embed')
    test_x = np.array([i for i in test.dt['phases_vectors'].values])

    test_x = StandardScaler().fit_transform(test_x)

    for i in range(300):
        os.mkdir(f'{dirname}/weights/{i}')
        model = phases.rank(X, X, epochs=epochs, dirname=f'{dirname}/weights/{i}')
        print("Predicting test set rankings...")
        pred_scores = model.predict(test_x)
        print("Calculating pairwise distances...")
        rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

        df = test.dt[['phases']]
        df = df.assign(rankings=rankings)

        df.to_csv(f"{dirname}/Unexplored_ternaries_{i}.csv")
