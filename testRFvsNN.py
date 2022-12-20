import os
import sys
import numpy as np
np.random.seed(1)
import pandas as pd
import tensorflow as tf
from Phase2Vec_simple import Phase2Vec
from utils import *
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import RocCurveDisplay as RCD
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # =============== CLASSIFICATION ================

#   prop = 'max energy gap'
#   Tc = 4.5

    prop = 'max Tc'
    Tc = 300
    epochs = 400
    dirname = 'scon_test'
    log = 'scon_class_log'

    #SCON
   #phases = Phase2Vec('', load_phase_vectors='DATA/combined_mpds_scon_Tc.pklone_hot_phases.pkl')
   #phases1 = Phase2Vec('', load_phase_vectors='DATA/combined_mpds_scon_Tc.pkl', atomic_mode='magpie')

    # MAGNET
    phases = Phase2Vec('', load_phase_vectors='DATA/mpds_magnet_CurieTc8.pklone_hot_phases.pkl')
    phases1 = Phase2Vec('', load_phase_vectors='DATA/mpds_magnet_CurieTc8.pklone_hot_phases.pkl', atomic_mode='magpie')

    #BG
   #phases = Phase2Vec('', load_phase_vectors='DATA/mpds_band_gap_8.pklone_hot_phases.pkl')
   #phases1 = Phase2Vec('', load_phase_vectors='DATA/mpds_band_gap_8.pklone_hot_phases.pkl', atomic_mode='magpie')

    X1 = np.array([i for i in phases.dt['onehot'].values])
    X2 = np.array([i for i in phases1.dt['phases_vectors'].values])
    y1 = np.where(phases.dt[f'{prop}'].values > Tc, 1, 0)
    y2 = np.where(phases1.dt[f'{prop}'].values > Tc, 1, 0)
    
    #   # 0.2 Test set to build predictions:
#    _, names_test, _, _ = train_test_split(phases.dt['phases'], y, test_size=0.2, random_state=0)
#   X1, X_test1, y1, y_true1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
#   X2, X_test2, y2, y_true2 = train_test_split(X2, y2, test_size=0.2, random_state=0)


    for k in [140]:
        kfold = KFold(n_splits=5, shuffle=False)
        accs, f1s, aucs = [],[],[]
        split = 1
        best_auc = 1
        for i, j in kfold.split(X1):
            input_x1, input_y1  = X1[i], y1[i]
            test_x1, y_true1 = X1[j], y1[j]
            input_x2, input_y2  = X2[i], y2[i]
            test_x2, y_true2 = X2[j], y2[j]
     
            # Compare
            #model1, represent = phases.classifier(input_x1, input_y1, epochs=epochs, k_features=k)
            model2 = phases1.rf_classifier(input_x2, input_y2)

            #y1_out = np.where(model1.predict(test_x1)[:,1] > 0.5, 1, 0)
            #y2_out = np.where(model2.predict(test_x2) > 0.5, 1, 0)
            y2_out = model2.predict_proba(test_x2)[:,1]
            y_pred = np.where(y2_out > 0.5, 1, 0)

            auc = roc_auc_score(y_true2, y_pred)

            if auc < best_auc:
                print('AUC', auc)
                best_auc = auc
                pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('mag_rf_classifier_roc.csv', index=False)
               # pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('scon_rf_classifier_roc.csv', index=False)
               # pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('bg_rf_classifier_roc.csv', index=False)

            #f1s.append(f1_score(y_true1, y1_out))

            # Plot ROC curve
            #RCD.from_predictions(y_true1, y1_out)
            #RCD.from_predictions(y_true2, y2_out)
            #plt.show()


            #print(f'SPLIT {split} NN acc, f1, roc_auc:', accuracy_score(y_true1, y1_out), f1_score(y_true1, y1_out), roc_auc_score(y_true1, y1_out))
            #print('RF acc, f1, roc_auc:', accuracy_score(y_true2, y2_out), f1_score(y_true2, y2_out), roc_auc_score(y_true2, y2_out))
            split += 1
            #pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('bg_rf_classifier_roc.csv', index=False)
            #pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('mag_rf_classifier_roc.csv', index=False)
            #pd.DataFrame({'y_true': y_true2, 'y_predict': y2_out}).to_csv('scon_rf_classifier_roc.csv', index=False)




        #print(k, np.mean(accs), np.mean(f1s))
