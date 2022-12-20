import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from Phase2Vec_simple import Phase2Vec
from sklearn.ensemble import RandomForestClassifier as RFC
from utils import *
import matplotlib.pyplot as plt


dirname =  'results'                          # directory where results are to be saved
savefile = 'test.csv'                 
# superconducting materials
input_file = 'DATA/combined_mpds_scon_Tc.pkl' # path to the phase fields input data file:
# magnetic materials
#input_file = 'DATA/mpds_magnet_CurieTc.csv'  # path to the phase fields input data file:
#input_file = 'DATA/mpds_band_gap_8.pkl'      # path to the phase fields input data file:
                                              # data should be prepared such that header is
                                              # "phases,max Tc"
                                              # and each line has a phase_field - max_value pair, e.g.:
                                              #  "Ir U Zn,9.2"
Tc = 10                                       # threshold value to divide two classes of performance
test_data = 'DATA/Ternary_phase_fields.pkl'   # path to the unexplored phase fields data file
                                              # which is a column of phase fields, e.g.:
                                              #  "Ca F Zr Br O"

if __name__ == '__main__':
    # Check files exist:
    for f in [dirname, input_file, test_data]:
        if not os.path.exists(f'{f}'):
            print(f'{f}: path does not exist. Exiting. ')
            sys.exit(0)


    # =============== CLASSIFICATION ================
 
    epochs = 10

    # Training set - all phase fields
    if os.path.exists(f'{input_file}one_hot_phases.pkl'):
        phases = Phase2Vec('', load_phase_vectors=f'{input_file}one_hot_phases.pkl')
    else:
        phases = Phase2Vec('', load_phase_vectors=f'{input_file}')
    X = np.array([i for i in phases.dt['onehot'].values])
    y = phases.dt['max Tc'].values
    y_class = np.where(y > Tc, 1, 0)

    # Predict unexplored
    if os.path.exists(f'{test_data}one_hot_phases.pkl'):
        test = Phase2Vec("", load_phase_vectors=f'{test_data}one_hot_phases.pkl', maxlength=phases.maxlength)
    else:
        test = Phase2Vec("", load_phase_vectors=f'{test_data}', maxlength=phases.maxlength)

    # Choose a size of an ensemble of models
    ensemble_n = 1
    for i in range(ensemble_n): 
        # =============== MODEL CLASSIFY  ================
        model, represent = phases.classifier(X, y_class, epochs=epochs)

        # =============== TEST =============================
        print("Predicting classification for unexplored phase fields...")
        test_x = np.array([i for i in test.dt['onehot'].values])
        y_test = model.predict(test_x)
        y_test = y_test[:,1] # look at the '1' class probability
        dttest = test.dt.assign(probability_high_performing_class=y_test)
        dttest = dttest[['phases','probability_high_performing_class']]
        print('Classification completed.')
        print(dttest.head())

        # =============== MODEL REGRESS  ================
        print("Predicting maximum value of the property ...")
        model, input_model = phases.regressor(X, y, k_features=120, epochs=epochs, batch_size=40, verbose=0)
        y_predict = model.predict(test_x).flatten()
        dttest = dttest.assign(predicted_max_value=y_predict)
        print(dttest.head())
        print('Regression completed.')

        # =============== MODEL RANKING  ================
        print("Predicting ranking ...")
        input_x = input_model(X)
        ranking_model = phases.rank(input_x, epochs=epochs, verbose=0)
        output_x = ranking_model.predict(input_x)
        train_scores = pairwise_distances_no_broadcast(input_x, output_x)
        test_x = input_model(test_x)
        test_scores = pairwise_distances_no_broadcast(test_x, ranking_model.predict(test_x))
        dttest = dttest.assign(chemical_novelty=test_scores)
        print(dttest.head())
        print('Ranking completed.')

        dttest.to_csv(f'{dirname}/{savefile}', index=None)

        # test
        plt.hist(train_scores,100, label='training data', alpha=1)
        plt.hist(test_scores,20, label='test data', alpha=0.3)
        plt.xlabel('Chemical novelty ranking, RE')
        plt.legend()
        plt.show()
