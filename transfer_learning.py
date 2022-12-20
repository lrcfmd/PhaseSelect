import sys
import numpy as np
np.random.seed(1)
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Phase2Vec_simple import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from Models.Model import *
from Models.AtomicModel_transfer import Endtoend
from utils import *
import matplotlib.pyplot as plt


def plot_r2(y_test, y_predict, prop='energy gap, eV'):
    error = [abs(t-p) for t,p in zip(y_test, y_predict)]
    r2 = r2_score(y_test, y_predict)
    plt.scatter(y_test, y_predict, c = error, label=fr'r$^2$={r2}')
    plt.xlabel(f'True {prop}')
    plt.ylabel(f'Predicted {prop}')
    lims = [max(y_test), min(y_test)]
    plt.plot(lims, lims, label=r'r$^2$ = 1')
    plt.legend()
    plt.show()
    return r2

if __name__ == '__main__':

    # 1. TRAIN on BG dataset
    # 2. Freeze the weights
    # 3. TRANSFER LEARN - predict for Magnetic and SCON dataset
    # 4. Fine tune the model?

    epochs = 100

    # Specify data files
    dirname = 'TL_regress'
    log = 'tl_regression_log'

    # max elements = 8 
    #training_data = 'DATA/mpds_band_gap_8.pklone_hot_phases.pkl'
    data2 = 'DATA/mpds_magnet_CurieTc8.pklone_hot_phases.pkl'
#    data3 = 'DATA/combined_mpds_scon_Tc.pklone_hot_phases.pkl'

    # 1. NN REGRESSION
    #phases = Phase2Vec('', load_phase_vectors=training_data, dirname=dirname, atomic_mode='envs')
    phases2 = Phase2Vec('', load_phase_vectors=data2, dirname=dirname, atomic_mode='envs')
    #phases3 = Phase2Vec('', load_phase_vectors=data3, dirname=dirname, atomic_mode='envs')
    
    # DATA set - all phase fields (vectorized)
    #X = np.array([i for i in phases.dt['onehot'].values])
    X2 = np.array([i for i in phases2.dt['onehot'].values])
    #X3 = np.array([i for i in phases3.dt['onehot'].values])
    #y = phases.dt['max energy gap'].values
    y2 = phases2.dt['max Tc'].values
    #y3 = phases3.dt['max Tc'].values

    # Transform Y:
    #y2 = np.log1p(y2)
    #y3 = np.log1p(y3)

    # 0.2 Test train split
    #_, names_test, _, _ = train_test_split(phases.dt['phases'], y, test_size=0.2, random_state=0)
    _, names_test2, _, _ = train_test_split(phases2.dt['phases'], y2, test_size=0.2, random_state=0)
    #_, names_test3, _, _ = train_test_split(phases3.dt['phases'], y3, test_size=0.2, random_state=0)
    #X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X2, X_test2, y2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=0)
    #X3, X_test3, y3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=0)

    # Prediction and plotting / on original data:
    model1, inner_model = phases2.regressor(X2, y2, X_test2, y_test2, k_features=120, epochs=1000, batch_size=512)
    y_predict = model1.predict(X_test2).flatten()
    r2 = plot_r2(y_test2, y_predict)
    
    pd.DataFrame({'phases': names_test2, 'true Tc': y_test2, 'predicted Tc': y_predict}).to_csv(f'{dirname}/mag_regressor_{r2}.csv', index=False)

###########################################

    # 2. FIX MODEL
    def fix_model(inner_model, lr=1e-3, step=50):
        inner_model.trainable = False
        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=lr,
                decay_steps=step,
                decay_rate=1)
        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        model = tf.keras.Sequential([inner_model, tf.keras.layers.Dense(1)])
        model.compile(optimizer=optim, loss="mean_absolute_error")
        return model

    # 3. TRANSFER  learn - Prediction and plotting / on data2,3:

    # a. dummy:
    #y_predict2 = model1.predict(X_test2).flatten()
    #plot_r2(y_test2, y_predict2, 'dummy, K')
    #y_predict3 = model1.predict(X_test3).flatten()
    #plot_r2(y_test3, y_predict3, 'dummy, K')

    # b. informed:
   #model2 = fix_model(inner_model, lr=1e-3)
   #early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
   #h2 = model2.fit(X2, y2,
   #              batch_size=40,
   #              epochs=300,
   #              callbacks=[early],
   #              validation_split=0.2,
   #              validation_data=[X_test2, y_test2],
   #              verbose=1,
   #              )
   #phases.plot_history(h2.history)
   #y_predict2 = model2.predict(X_test2).flatten()

    # b. informed:
   #model3 = fix_model(inner_model)
   #early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
   #h3 = model3.fit(X3, y3,
   #              batch_size=40,
   #              epochs=1000,
   #              callbacks=[early],
   #              #validation_split=0.2,
   #              validation_data=[X_test3, y_test3],
   #              verbose=1,
   #              )
   #phases2.plot_history(h3.history)
   #y_predict3 = model3.predict(X_test3).flatten()
   #plot_r2(y_test3, y_predict3, 'Tc, K')
    # save
    #pd.DataFrame({'phases': names_test, 'true bg eV': y_test, 'predicted bg eV': y_predict}).to_csv(f'{dirname}/bg_regressor_k120.csv', index=False)
    #pd.DataFrame({'phases': names_test2, 'true Tc': y_test2, 'predicted Tc': y_predict}).to_csv(f'{dirname}/mag_regressor_{r2}.csv', index=False)
    #pd.DataFrame({'phases': names_test3, 'true Tc K': y_test3, 'predicted Tc K': y_predict3}).to_csv(f'{dirname}/scon_regressor_transfer.csv', index=False)
