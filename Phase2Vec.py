import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Classification.Model import * 
from Classification.AtomicModel import Endtoend
from utils import *

class Phase2Vec():
    def __init__(self, dbfile, epochs=10, load_phase_vectors=None, maxlength=None, dirname='AtomicTraining', 
                 atomic_mode='envs', natom=None, attention=False, ifaugment=False):
       
        if load_phase_vectors:
           self.read_phase_vectors(load_phase_vectors, natom)
           self.datafile_name = load_phase_vectors
        else:
           self.read_db(dbfile)

        if ifaugment:
            self.dt = augment_permute(self.dt)

        self.attention = attention
        self.maxlength = maxlength 
        self.get_atomvec(atomic_mode)

        if 'onehot' not in self.dt.columns and 'phases_vectors' not in self.dt.columns:
           self.dt = self.cleanphase()
           self.phase2vec(atomic_mode)
       
    def phase2vec(self, mode):
        """ mode envs: end-to-end training of the atomic vectors and phase vectors (classification)
            mode embed: precalculated atom2vec, phase2vec from dictionary (AE ranking) """
        if mode == 'envs':
            self.phasehot()
        else:
            print(self.maxlength)
            if self.attention:
                self.dt, self.maxlength = embedding_attention(self.dt, self.atom2vec, self.maxlength) 
            else:
                self.dt, self.maxlength = embedding(self.dt, self.atom2vec, self.maxlength) 

    def read_phase_vectors(self, load_phase_vectors, natom=None):
        """ Note: Sampling shuffles data, may result in varying results
            during k-fold cross validation """ 
        print(f"Reading from {load_phase_vectors}")
        try: 
            self.dt = pd.read_pickle(load_phase_vectors).sample(frac=1)
        except Exception:
            self.dt = pd.read_csv(load_phase_vectors).sample(frac=1)
        if natom:
            print(f"Selecting phases with {natom} elements")
            print(self.dt.shape)
            self.dt['natom'] = self.dt['phases'].apply(lambda x: len(x.split()))
            self.dt = self.dt[self.dt['natom'] == natom]
            print(self.dt.shape)

    def read_db(self, dbfile):
        self.dt = parse_phases(dbfile) 

    def get_atomvec(self, mode='envs'):
        if mode == 'magpie':
            with open('atom2vec_dic.pickle', 'rb') as handle:
                self.atom2vec = pickle.load(handle)
            self.atoms = list(self.atom2vec.keys())
            print("Created precalc. Atom2Vec dictionary, N features:", len(list(self.atom2vec.values())[0]))
        else:
            atoms = Atom2Vec("Atom2Vec/string.json", 40, mode=mode)
            self.atoms = np.array(atoms.elements)
            if mode=='envs':
                self.envs_mat = atoms.envs_mat
            else:
                self.atom2vec = {ind: vec for ind, vec in zip(atoms.elements, atoms.atoms_vec)}
                print("Created precalc. Atom2Vec dictionary, N features:", len(list(self.atom2vec.values())[0]))

    def cleanphase(self):
        """ Remove elements not vectorised"""
        print("Checking all atoms are vectorized...")       
        setatoms = set(self.atoms)

        def exl(x):
            return set(x.split()).issubset(setatoms)

        self.dt['missing'] = list(map(exl, self.dt['phases'].values))
        #self.dt[self.dt['missing'] != 0].to_csv('missing_elements.csv', index=False)
        self.dt = self.dt[self.dt['missing']]

        return self.dt.drop(columns='missing')

    def phasehot(self):
        """ create onehot encoding for phase fields: yes/no (1/0) atom
            this is further used during end2end training to form phase vectors
            from atomic vectors (latent, first stage of the training)
            by matmul: phasehot @ atom2vec """
        print("One-hot encoding of the phase fields ...")
        print(f"Vector lenght: {self.maxlength}")
        atoms = tuple(self.atoms)
        L = len(atoms)
        if self.maxlength is None:
            self.maxlength = max(self.dt['phases'].apply(lambda x: len(x.split())))
        print (self.maxlength)
        def onehot(x):
             p = list(map(lambda el: atoms.index(el), x.split()))
             one = np.zeros((self.maxlength, L))
             for i, el in enumerate(p):
                 one[i,el] = 1
             return one

        self.dt['onehot'] = self.dt['phases'].apply(lambda x: onehot(x))

        self.dt.to_csv(self.datafile_name + 'one_hot_phases.csv')
        self.dt.to_pickle(self.datafile_name + 'one_hot_phases.pkl')

        print(self.dt.head())

    def rank(self, input_x, valids=None, epochs=10, dirname='Training_ranking'):
        """ Use AE for ranking wrt reconstruction error """
        print("INPUT fo RANK")
        print(type(input_x))
        print(input_x.shape)
        print(input_x[0])

        k_features = len(list(self.atom2vec.values())[0])

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)

        if self.attention:
            print('Ranking with attention')
            model, get_attention = RankingAE_attention(k_features)(input_x)
        else:
            model = RankingAE()(input_x)

        model.compile(optimizer='adam')

        history = model.fit(input_x, input_x,
                  batch_size=256, 
                  epochs=epochs,
                  callbacks=[early],
                  validation_split=0.8,
                  )

        with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        #Phase.plot_history(history, dirname)

        # Calculate training set scores:
        #output_x = model.predict(input_x)
        #rankings = pairwise_distances_no_broadcast(input_x, output_x)

        if self.attention: return model, history, get_attention
        else: return model, history
 
    def classifier(self, input_x, input_y, valids=None, epochs=10, dirname='Training', k_features=40):

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate 

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7, restore_best_weights=True)

        model, att_scores = Endtoend(k_features)(input_x, self.envs_mat, attention=True)
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) # or accuracy

        history = model.fit(input_x, input_y,
                  batch_size=256, 
                  epochs=epochs,
                  callbacks=[create_callback(dirname), early],
                  validation_data=valids,
                  class_weight=class_weights(input_y)
                  )
        with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        #Phase.plot_history(history, dirname)

        return model, att_scores

    def classify_validate(self, X, y, n_splits=5, epochs=10, Tc=10, dirname='endTraining', log='log', prop='max Tc', k_features=50):
        """ Run k-fold validation """
        accuracy, f1score = [], []
        
        kfold = KFold(n_splits=n_splits, shuffle=False)

        it = 0
        for i, j in kfold.split(X): 
            it += 1
            input_x, test_x, input_y, test_y, dftest = split(X, y, self.dt, i, j, prop=prop)
            
            model, _ = self.classifier(input_x, input_y, (test_x, test_y), epochs, dirname=f'{dirname}/fold2_{str(it)}', k_features=k_features)

            acc, sc = Phase.predict_class(model, test_x, test_y, dftest, Tc=Tc, dirname=f'{dirname}/fold2_{str(it)}', prop=prop)
            accuracy.append(acc)
            f1score.append(sc)

        print(f"Average accuracy: {np.average(accuracy)}", file=open(f'{dirname}/{log}', 'a+'))
        print(f"Average f1 score: {np.average(f1score)}", file=open(f'{dirname}/{log}', 'a+'))


    def classify_once(self, X, y, epochs=10, Tc=10, dirname='Test', log='log', prop='max Tc', k_features=40):
        """ Classification with a random 20% validation set """

        cut = int(len(y)*0.8)
        input_x, test_x = X[:cut,:], X[cut:,:]
        input_y, test_y = y[:cut], y[cut:]
        dtest = pd.DataFrame({'phases': self.dt['phases'].values[cut:],
                           f'{prop}': self.dt[prop].values[cut:]})

        model, _ = self.classifier(input_x, input_y, (test_x, test_y), epochs, dirname=dirname, k_features=k_features)
        acc, sc = Phase.predict_class(model, test_x, test_y, dtest, Tc=Tc, dirname=dirname, prop=prop)
        print(f"accuracy: {acc}", file=open(f'{dirname}/{log}', 'a'))
        print(f"f1 score: {sc}", file=open(f'{dirname}/{log}', 'a'))

    def predict_ranking(self, epochs=10, dirname='Ranking', log='log'): 
        """ Use precalculated atom2vec vectors to build phase vectors
            Run AE to get RE scores """

        X = np.array([i for i in self.dt['phases_vectors'].values])

        cut = int(len(X)*0.80)
        #cut = clean_cut(self.dt['phases'], cut)
        
        input_x, test_x = X[:cut], X[cut:]
        dtest = pd.DataFrame({'phases': self.dt['phases'].values[cut:],
                           'max Tc': self.dt['max Tc'].values[cut:]})

        model, attention = self.rank(input_x, test_x, epochs, dirname=dirname)
        normalize = StandardScaler().fit(test_x)
        test_x = normalize.transform(test_x)

        # Predict on X and return the reconstruction errors
        pred_scores = model.predict(test_x)
        atte_scores = attention.predict(test_x)
        rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

        df = dtest.assign(rankings=rankings)
        normalized_df=(df['rankings']-df['rankings'].min())/(df['rankings'].max() - df['rankings'].min())
 
        df['rankings'] = df['rankings'].apply(lambda x: round(x,3))
        df['norm_score'] = normalized_df
        df['norm_score'] = df['norm_score'].apply(lambda x: round(x,3))
        df = df.sort_values(['rankings'])

        df.to_csv('Ranked_test.csv', index=False)

        df['attention_scores'] 

if __name__ == '__main__':

    # =============== CLASSIFICATION ================
 
    # Start new calculations
    #phases = Phase2Vec('DB/Supercon_data.csv')

    Tc = 300
    epochs=100

    dirname='Magnet_unexplored_ranking'
    log='Magnet_unexplored_log'
    #log='Magnet_classification_accuracy'
    #log='End_to_end_training_combined_data_results.txt'
 
    # load phase vectors if precalculated
    #phases = Phase2Vec('DB/Supercon_data.csv', load_phase_vectors='DB/DATA/combined_mpds_scon_Tc.pkl', atomic_mode='read')
    #phases = Phase2Vec('DB/Supercon_data.csv', load_phase_vectors='DB/DATA/SuperCon_MPDS_one_hot_phases.pkl')


#   # MODEL CLASSIFY
#   phases = Phase2Vec('', load_phase_vectors='DB/DATA/mpds_magnet_CurieTc.csv')
#   # Training - all phase fields (vectorized)
#   X = np.array([i for i in phases.dt['onehot'].values]) 
#   y = np.where(phases.dt['max Tc'].values > Tc, 1, 0)
#
#   # Testing - unexplored phase fields
#   test = Phase2Vec('', load_phase_vectors='DB/DATA/magnetic_candidates_scores.csv', maxlength=phases.maxlength)
#   test_x = np.array([i for i in test.dt['onehot'].values])
#    
#   model = phases.classifier(X, y, epochs=epochs, dirname=dirname)
#   y_test = model.predict(test_x)
#   y_test = y_test[:,1] # look at the '1' class
#
#   dttest = test.dt.assign(probability_Tc300=y_test)
#   dttest.to_csv('Magnetic_unexplored_scores.csv') 
#
#   sys.exit()


    # MODEL AE RANKING
    phases = Phase2Vec('', load_phase_vectors='DB/DATA/mpds_magnet_CurieTc.csv',atomic_mode='read')
    X = np.array([i for i in phases.dt['phases_vectors'].values])

    test = Phase2Vec('', load_phase_vectors='Magnetic_unexplored_scores.csv', maxlength=phases.maxlength, atomic_mode='read')
    test_x = np.array([i for i in test.dt['phases_vectors'].values])

    model = phases.rank(X, X, epochs=epochs, dirname=dirname)
    test_x = StandardScaler().fit_transform(test_x)
    pred_scores = model.predict(test_x)
    rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

    df = test.dt.assign(rankings=rankings)
    normalized_df=(df['rankings']-df['rankings'].min())/(df['rankings'].max() - df['rankings'].min())

    df['rankings'] = df['rankings'].apply(lambda x: round(x,3))
    df['norm_score'] = normalized_df
    df['norm_score'] = df['norm_score'].apply(lambda x: round(x,3))
    df = df.sort_values(['rankings'])
    
    df.to_csv('Magnetic_unexplored_scores_rankings.csv', index=None) 

