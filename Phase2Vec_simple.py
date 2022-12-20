import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Atom2Vec.EnvMatrix import EnvsMat
from Models.Model import split, RankingAE
from Models.AtomicModel_transfer import Endtoend as Endtoend_transfer
from Models.AtomicModel import Endtoend
from periodic_table import ELEMENTS
import utils
import functools
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization as BO

class Phase2Vec():
    def __init__(self, dbfile, epochs=300, load_phase_vectors=None, maxlength=None, dirname='AtomicTraining', 
                 atomic_mode='envs', attention=False, ifaugment=False, embeddings=''):
      
        self.epochs = epochs
        self.dirname = dirname
        self.attention = attention
        self.datafile_name = load_phase_vectors
        self.atomic_mode = atomic_mode
        self.embeddings = embeddings

        self.atoms = ELEMENTS                         
        self.maxlength = maxlength         # Max number of elements in a phase field in a dataset
        self.dt = self.read_phases()       # DataFrame with columns e.g. [phases, onehot, max Tc]
        self.atom2vec = self.get_atomvec() # Read vectors (Magpie) or atomic enviroments (Atom2Vec)

        if self.start_anew():              # One-hot encode phase field for matmul with atomic vectors 
           self.dt = self.cleanphase()
           self.phase2vec()
       
    def start_anew(self):
        if 'onehot' not in self.dt.columns or  \
        (self.atomic_mode=='magpie' and 'phases_vectors' not in self.dt.columns):
            return True
        else:
            return False

    def read_phases(self):
        """ Note: Sampling shuffles data, may result in varying results
        during k-fold cross validation """
        print(f"Reading phase fields from {self.datafile_name}")
        try: 
            dt = pd.read_pickle(self.datafile_name).sample(frac=1)
        except Exception:
            dt = pd.read_csv(self.datafile_name)
        return dt

    def read_db(self, dbfile):
        self.dt = parse_phases(dbfile) 

    def get_atomvec(self):
        # 1. magpie vectors
        if self.atomic_mode == 'magpie':
            with open('Atom2Vec/magpie_atomic_features.pkl', 'rb') as handle:
                atom2vec = pickle.load(handle)
            self.atoms = list(atom2vec.keys())
            print("Created precalc. Atom2Vec dictionary, N features:", atom2vec.shape)
        # 2. calculate vectors from atomic environments during training
        elif self.atomic_mode == 'envs':
            if os.path.exists('ENVS_MAT.pkl'):
                print(f'Reading atomic environment from ENVS_MAT.pkl')
                with open('ENVS_MAT.pkl', 'rb') as f:
                    self.envs_mat = pickle.load(f)
            else:
                envs = EnvsMat("Atom2Vec/string.json")
                self.envs_mat = envs.envs_mat
                with open('ENVS_MAT.pkl', 'wb') as f:
                    pickle.dump(self.envs_mat, f)
            atom2vec = {atom: [] for atom in self.atoms}
        # 3. use precalculated atomic vectors from previous trainings
            if bool(self.embeddings):
                self.embeddings = self.read_embeddings()
                atom2vec = {atom: vector for atom, vector in zip(self.atoms, self.embeddings)}
        return atom2vec

    def read_embeddings(self):
        """ Read embeddings file if precalculated """
        print(f'Reading precalculated atomic embeddings from {self.embeddings}')
        with open(f'{self.embeddings}', 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    def cleanphase(self):
        """ Remove elements not vectorised"""
        print("Checking all atoms are vectorized...")       
        setatoms = set(self.atoms)

        def exl(x):
            return set(x.split()).issubset(setatoms)

        self.dt['missing'] = list(map(exl, self.dt['phases'].values))
        # Save missing elements for analysis
        #self.dt[self.dt['missing'] != 0].to_csv('missing_elements.csv', index=False)
        self.dt = self.dt[self.dt['missing']]

        return self.dt.drop(columns='missing')

    def phase2vec(self):
        """ mode 'envs':   atomic vectors are learnt from AE as latent vectors in Atom2Vec fashion 
            mode 'magpie': atomic vectors are built from magpie features"""
        if self.atomic_mode == 'envs':
            self.phasehot()
        elif self.atomic_mode == 'magpie':
            if self.attention:
                self.dt, self.maxlength = utils.embedding_attention(self.dt, self.atom2vec, self.maxlength) 
            else:
                self.dt, self.maxlength = utils.embedding(self.dt, self.atom2vec, self.maxlength) 

    def phasehot(self):
        """ create onehot encoding for phase fields: yes/no (1/0) atom
            this is further used during end2end training to form phase vectors
            from atomic vectors (latent, first stage of the training)
            by matmul: phasehot @ atom2vec """
        print("One-hot encoding of the phase fields ...")
        print(f"Vector length: {self.maxlength}")
        atoms = tuple(self.atoms)
        L = len(atoms)
        if self.maxlength is None:
            self.maxlength = max(self.dt['phases'].apply(lambda x: len(x.split())))

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

    def rank(self, input_x, valids=None, epochs=100, dirname='Training_ranking', verbose=0):
        """ Use AE for ranking wrt reconstruction error """

        k_features = len(list(self.atom2vec.values())[0])

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)

        model = RankingAE()(input_x)

        model.compile(optimizer='adam')

        history = model.fit(input_x, input_x,
                  batch_size=256, 
                  epochs=epochs,
                  callbacks=[early],
                  validation_split=0.2,
                  verbose=verbose
                  )

        #with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
        #    pickle.dump(history.history, file_pi)

        #self.plot_history(history, dirname)

        # Calculate training set scores:
        #output_x = model.predict(input_x)
        #rankings = pairwise_distances_no_broadcast(input_x, output_x)
        return model

    def plot_history(self, history, prop='loss', value='Band gap, eV'):
        plt.plot(history[f'{prop}'], label='train')
        plt.plot(history[f'val_{prop}'], label='test')
        plt.xlabel('Epoch')
        plt.ylabel(f'MAE [{value}')
        plt.legend()
        plt.show()

    def rf_classifier(self, X, y):
        cls = RFC(max_depth=4,random_state=0)
        cls.fit(X, y)
        return cls

    def rf_regressor(self, X, y, **kwargs):
        reg = RFR(max_depth=4,random_state=0)
        reg.fit(X, y)
        return reg

    def regressor(self, X, y, test_x=None, test_y=None, epochs=1000, k_features=120, hidden_layer=500, batch_size=40, 
            transfer=False, predict_from=False, filename=None, validate=False, verbose=0):

        if transfer:
            unfix = False
        else:
            unfix = True
        
        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-3,
                decay_steps=100,
                decay_rate=1)

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        # MODEL
        input_model, inner_model, full_model = Endtoend_transfer(k_features, 
                hidden_layer, model_type='reg', unfix=unfix)(X, self.envs_mat, self.attention)
        # Transfer weights from other model
        if transfer:
            print(f'Loading weights from {transfer}')
            inner_model.load_weights(f'{transfer}')
            inner_model.trainable = False
            full_model = tf.keras.Sequential([inner_model, tf.keras.layers.Dense(1)])

        full_model.compile(optimizer=optim, loss="mean_absolute_error")
        dirname = self.dirname

        if predict_from:
            full_model.load_weights(f'{predict_from}')
            return full_model

        history = full_model.fit(X, y,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[early],
                  validation_split=0.2,
                  #validation_data=[test_x, test_y],
                  verbose=verbose,
                  )

#       historyname = f'k{k_features}_nodes{hidden_layer}_bsz{batch_size}_history'
#       with open(f'{dirname}/{historyname}', 'wb') as file_pi:
#           pickle.dump(history.history, file_pi)

        self.plot_history(history.history)

        # save full model weights
        # weights = full_model.get_weights()
        # filename = f'k{k_features}_nodes{hidden_layer}_bsz{batch_size}.hdf5'
        # print(f'Collected full model weights: {dirname}/full_model_weights.hdf5')
        # full_model.save_weights(f'{dirname}/{filename}')

        # weights = inner_model.get_weights()
        # print('Collected inner model weights arrays:', len(weights))
        # inner_model.save_weights(f'{dirname}/inner_model_weights_k{k_features}_nodes{hidden_layer}_fromTransfer.hdf5')

        if transfer:
            return full_model, inner_model
        elif validate:
            return full_model
        else:
            return full_model, input_model

     
    def validate(self, model, X, y, loss='mae', *args, **kwargs):
        """ validate classifier or regressor in 5-fold """

        kfold = KFold(n_splits=5, shuffle=False)
        errors = []
        n = 1
        for i, j in kfold.split(X):
            input_x, input_y  = X[i], y[i]
            test_x, test_y = X[j], y[j]
            m = model(input_x, input_y, test_x=test_x, test_y=test_y, validate=True, **kwargs)
            y_predict = m.predict(test_x).flatten()

            if loss=='r2':
                error = r2_score(test_y, y_predict)
            else:
                error = mean_absolute_error(test_y, y_predict)
            error = round(error, 2)
            print(f'K SPLIT: {n}--------- ERROR: {loss}: {error}')

            pd.DataFrame({'y_true': test_y, 'y_predicted': y_predict}).to_csv(f'{self.dirname}/r2_test_k{n}_loss{error}.csv')
            n += 1
            errors.append(error)

        return errors
 
    def classifier(self, input_x, input_y, valids=None, epochs=10, k_features=120, hidden_layer=500, batch_size=256):

        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=-0.9)  # increasing rate 

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=17, restore_best_weights=True)

        model, represent = Endtoend(k_features, hidden_layer)(input_x, self.envs_mat, attention=True)
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) # or loss

        dirname = self.dirname

        history = model.fit(input_x, input_y,
                  batch_size=batch_size, 
                  epochs=epochs,
                  callbacks=[early],
                  #validation_split=0.2,
                  validation_data=valids,
                  class_weight=utils.class_weights(input_y),
                  verbose=0,
                  )
#       with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
#           pickle.dump(history.history, file_pi)

        #self.plot_history(history.history)

        return model, represent

    def hyperopt(self, X, y):
        """ use sets of k_features, hidden_layer to find best average accuracy in CV """

        space = {'k_features': (20,500), 'hidden_layer': (20,500), 'batch_size':(20,400)}
        space = {'k_features': (80,200), 'hidden_layer': (200,500), 'batch_size':(20,500)}

        def discretize(k, h, b):
            k = round(k/400,0) * 40
            h = round(h/100,0) * 100
            b = round(b/20,0) * 20
            return k, h, b

        def boloss(k_features, hidden_layer, batch_size):
           #loss = self.classify_validate(X, y,
           #        n_splits=3,
           #        k_features=int(k_features), 
           #        hidden_layer=int(hidden_layer), 
           #        batch_size=int(batch_size)) 
            
            k, h, b = discretize(k_features, hidden_layer, batch_size)

            loss = self.regressor_validate(X, y,
                    n_splits=5,
                    k_features=int(k),
                    hidden_layer=int(h),
                    batch_size=int(b))

            return loss

        run_bo = BO(boloss, space)
        run_bo.maximize(init_points=10, n_iter=30)

        return run_bo.max['params']


    def classify_validate(self, X, y, n_splits=3, k_features=40, hidden_layer=80, batch_size=250):
        """ Run k-fold validation """
        accuracy, f1score = [], []
        
        kfold = KFold(n_splits=n_splits, shuffle=False) #random_state=42)

        for i, j in kfold.split(X): 
            input_x, test_x, input_y, test_y, dftest = split(X, y, self.dt, i, j)
            
            model, _,  history = self.classifier(input_x, input_y, 
                (test_x, test_y), k_features=k_features, hidden_layer=hidden_layer, batch_size=batch_size)

            acc = history.history['val_accuracy'][-1]
            accuracy.append(acc)

            #print(f"Accuracy: {acc}", file=open(f'{self.dirname}/CV_mag_class.txt', 'a+'))

       # print(f"Average accuracy: {np.average(accuracy)}", file=open(f'{self.dirname}/CV_mag_class.txt', 'a+'))
       # for hyperopt
        return np.mean(accuracy)


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

