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

    Tc = 10
    epochs = 100
    dirname = 'MAGNETIC/Ranking_quaternaries_single_run'

    # =============== AE RANKING  ================

    # load phase fields of phase vectors if precalculated
    #phases = Phase2Vec('', load_phase_vectors='DB/DATA/mpds_magnet_8_CurieTc.csv', atomic_mode='embed')
    phases = Phase2Vec('', load_phase_vectors='DB/ICSD/icsd_phases_8.csv', atomic_mode='embed')
    X = np.array([i for i in phases.dt['phases_vectors'].values if len(i)])

    print("SIZE OF ICSD phases: ", phases.dt.shape)

    #test = Phase2Vec("", load_phase_vectors='Unexplored_quaternaries.pkl', atomic_mode='embed',maxlength=phases.maxlength)
    test = Phase2Vec("", load_phase_vectors='DB/DATA/magnetic_candidates.csv', atomic_mode='embed',maxlength=phases.maxlength)
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

#       print("Normalizing the scores...")
#       normalized_df=(df['rankings']-df['rankings'].min())/(df['rankings'].max() - df['rankings'].min())
#        df['norm_score'] = normalized_df
#        df['norm_score'] = list(map(rr, df['norm_score']))
#        df = df.sort_values(['phases'])
        #df.to_csv(f"Rankings_Magnet_quaternaries_reproduce/Unexplored_quaternaries_{i}.csv")
        df.to_csv(f"{dirname}/ranking_magnet_quaternaries_single_run.csv")

        # plot history
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        plt.show()


sys.exit(0)


# read precalculated df with classification Tc > 10K:
print("Reading the Tc > 10K classifications dataframe...")
clas = pd.read_csv("Supercon_unexplored_ternaries_scores.csv")
clas = clas.sort_values(['phases'])
if df['phases'].values[0] == clas['phases'].values[0]:
    clas['norm_rank'] = df['norm_score'] 
    clas = clas.sort_values(['probability_Tc10K'],ascending=False)
 
    print("Writing results to Supercon_unexplored_scores_rankings.csv")
    clas.to_csv('Supercon_unexplored_ternaries_scores_rankings.csv', index=None)

else: 
    df.to_csv("Unexplored_ternaries_raw.csv")
    clas.to_csv('Supercon_unexplored_ternaries_scores_rankings.csv', index=None)


print('Plotting...')
import matplotlib.pyplot as plt
plt.scatter(clas['norm_rank'], clas['probability_Tc10K'], alpha=0.7)
plt.xlabel('AE normalized RE')
plt.ylabel("Probability Tc > 10K")
plt.show()
