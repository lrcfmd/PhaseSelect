import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from AtomicModel import * #Endtoend, RankingAE
from utils import * 
from post_process import *

class Phase2Vec():
    def __init__(self, dbfile, epochs=10, load_phases=None, maxlength=None, 
                dirname='AtomicTraining', mode='classify', atomfile="Atom2Vec/atoms_AE_vec_onehot.txt", natom=None):
        """ mode classify: end-to-end training of the atomic vectors and phase vectors classification
            mode rank: precalculated during 'classify' mode atom2vec, build phase2vec for AE ranking)
        """       

        if load_phases:
           self.read_phase_vectors(load_phases, natom)
           self.datafile_name = load_phases
        else:
           self.read_db(dbfile)

        self.maxlength = maxlength  
        self.get_atomvec(atomfile, mode)      # get atomic environments Atom2Vec 
                                              # (and dictionary of atomic vectors for 'rank' mode)
        self.phase2vec(mode)                  # get phase vectors 
       
    def phase2vec(self, mode):
        """ mode classify: end-to-end training of the atomic vectors and phase vectors classification
            mode rank: precalculated during 'classify' mode atom2vec, build phase2vec for AE ranking) """

        if mode == 'classify' and 'onehot' not in self.dt.columns:
            self.dt = self.cleanphase()
            self.phasehot()
        elif mode == 'rank':
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

    def get_atomvec(self, atomfile, mode):
        """ mode 'classify': read raw atomic environment matrix 
            mode 'rank': read precalculated atomic vectors
        """
        atoms = Atom2Vec("Atom2Vec/string.json", 20, atomvec_file=atomfile, mode=mode)
        self.atoms = np.array(atoms.elements)
        self.envs_mat = atoms.envs_mat

        if mode == 'rank':
            self.atom2vec = {ind: vec for ind, vec in zip(atoms.elements, atoms.atoms_vec)}
            print("Created precalc. Atom2Vec dictionary, N features:", len(list(self.atom2vec.values())[0]))

    def cleanphase(self):
        """ Remove elements not vectorised"""
        print("Checking all atoms are vectorized...")       
        setatoms = set(self.atoms)

        def exl(x):
            return set(x.split()).issubset(setatoms)

        self.dt['missing'] = list(map(exl, self.dt['phases'].values))
        self.dt = self.dt[self.dt['missing']]

        return self.dt.drop(columns='missing')

    def phasehot(self):
        """ create onehot encoding for phase fields: yes/no (1/0) atom.
            this is further used during end2end training to form phase vectors
            from atomic vectors (latent, first stage of the training)
            by matmul: phasehot @ atom2vec """
        print("One-hot encoding of the phase fields ...")

        atoms = tuple(self.atoms)
        length = len(atoms)

        if self.maxlength is None:
            self.maxlength = max(self.dt['phases'].apply(lambda x: len(x.split())))

        print ("Max length of a phase field:", self.maxlength)

        def onehot(x):
             p = list(map(lambda el: atoms.index(el), x.split()))
             one = np.zeros((self.maxlength, length))
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

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        model = RankingAE()(input_x)
        model.compile(optimizer='adam')

        history = model.fit(input_x, input_x,
                  batch_size=16, # 16 performs better
                  epochs=epochs,
                  callbacks=[create_callback(dirname), early],
                  validation_split=0.8,
                  )
        with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        return model   
 
    def classifier(self, input_x, input_y, valids=None, epochs=10, dirname='Training'):

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate 

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)

        model, embeddings = Endtoend()(input_x, self.envs_mat, attention=True)
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(input_x, input_y,
                  batch_size=16, # 16 performs better
                  epochs=epochs,
                  callbacks=[create_callback(dirname), early],
                  validation_data=valids,
                  class_weight=class_weights(input_y)
                  )
        with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        #Phase.plot_history(history, dirname)

        return model

    def classify_validate(self, X, y, kfold, epochs=10, Tc=10, dirname='endTraining', log='log'):
        """ Run k-fold validation """
        accuracy, f1score = [], []

        it = 0
        for i, j in kfold.split(X): 
            it += 1
            input_x, test_x, input_y, test_y, dftest = split(X, y, self.dt, i, j)
            
            model = self.classifier(input_x, input_y, (test_x, test_y), epochs, dirname=dirname+str(it))

            acc, sc = self.predict_class(model, test_x, test_y, dftest, Tc=Tc, dirname=dirname+str(it))
            accuracy.append(acc)
            f1score.append(sc)

        print(f"Average accuracy: {np.average(accuracy)}", file=open(log, 'a'))
        print(f"Average f1 score: {np.average(f1score)}", file=open(log, 'a'))


    def classify_once(self, X, y, epochs=10, Tc=10, dirname='Test', log='log'):
        """ Classification with a random 20% validation set """

        cut = int(len(y)*0.8)
        input_x, test_x = X[:cut,:], X[cut:,:]
        input_y, test_y = y[:cut], y[cut:]
        dtest = pd.DataFrame({'phases': self.dt['phases'].values[cut:],
                           'max Tc': self.dt['max Tc'].values[cut:]})

        model = self.classifier(input_x, input_y, (test_x, test_y), epochs, dirname=dirname)
        acc, sc = self.predict_class(model, test_x, test_y, dtest, Tc=Tc, dirname=dirname)
        print(f"accuracy: {acc}", file=open(log, 'a'))
        print(f"f1 score: {sc}", file=open(log, 'a'))

    def predict_ranking(self, epochs=10, dirname='Ranking', log='log'): 
        """ Use precalculated atom2vec vectors to build phase vectors
            Run AE to get RE scores """

        X = np.array([i for i in self.dt['phases_vectors'].values])

        cut = int(len(X)*0.80)
        
        input_x, test_x = X[:cut], X[cut:]
        dtest = pd.DataFrame({'phases': self.dt['phases'].values[cut:],
                           'max Tc': self.dt['max Tc'].values[cut:]})

        model = self.rank(input_x, test_x, epochs, dirname=dirname)
        normalize = StandardScaler().fit(test_x)
        test_x = normalize.transform(test_x)

        # Predict on X and return the reconstruction errors
        pred_scores = model.predict(test_x)
        rankings = pairwise_distances_no_broadcast(test_x, pred_scores)

        df = dtest.assign(rankings=rankings)
        normalized_df=(df['rankings']-df['rankings'].min())/(df['rankings'].max() - df['rankings'].min())
 
        df['rankings'] = df['rankings'].apply(lambda x: round(x,3))
        df['norm_score'] = normalized_df
        df['norm_score'] = df['norm_score'].apply(lambda x: round(x,3))
        df = df.sort_values(['rankings']) 

        df.to_csv('Ranked_test.csv', index=False)

    @staticmethod
    def predict_class(model, X_test, y_true,  dftest, Tc, dirname):
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,ConfusionMatrixDisplay

        y_test = model.predict(X_test)
        y_test = y_test[:,1] # look at the '1' class
        y_pred = np.where(y_test>=0.5, 1, 0)

        try:
           dftest = dftest.assign(prediction = y_pred)
           dftest = dftest.assign(probability_1 = y_test)
        except:
           print("Wrong Y_test size?")
           dftest = pd.DataFrame([dftest['phases'].values, dftest['max Tc'].values,\
                                  y_pred, y_test], \
                                  columns=['phases', 'Tc', 'prediction', 'probability_1'])

        dftest = reduce_duplicate(dftest)
        dftest.to_pickle(f'{dirname}/dtest.pkl')

        acc, sc = plot_confusion(dftest, Tc, f'{dirname}/2_class_prediction.doc')
        return acc, sc
