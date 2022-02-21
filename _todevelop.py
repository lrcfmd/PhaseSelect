import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Phase2Vec import Phase2Vec
from Atom2Vec.Atom2Vec_encoder import Atom2Vec
from AtomicModel import Endtoend
from utils import *

def get_phases(datafile=None, phases=None, mode='classify', maxlength=None):
    """ Reads datafile, trains or reads atomic features, vectorises phase fields.

    Parameters
    ----------
    datafile: str;           filename of database to process 
                             to aggregate compositions into phase fields
                             default: None
    load_phase_vectors: str; filename to read phase fields, or their vectors
                             default: None
    mode: str;               mode of use of atomic features - parameter is passed to Atom2Vec and End2End model
                             default: 'classify' - derive during End2End training from Atom2Vec enviroments
                             'rank' - re-use embeddings calculated in 'classify' mode
    maxlength: int;          maximum size of a phase field in the dataset
                             default: None
    """
    return Phase2Vec(dbfile=datafile, load_phases=phases, maxlength=maxlength, mode=mode)

def get_dataset(phases, parameter, t_type='max Tc', T=None, scaler=None):
    """ Selects data for training, X.
        If threshold T is specified, selects target values, y, from dataframe.

    Parameters
    ----------
    phases: Phase2Vec obj
    parameter: str; name of the column in phases.dt where phase vectors are stored
                    values:'onehot' or 'phase_vectors'
    t_type: str;    name of the column (property), which is used for data classification
                    default: 'max_Tc'
    T: float;       threshold value used for binary partitioning of the dataset  
    scaler: sklearn obj; perform normalisation on X data; default: None 

    """
    X = np.array([i for i in phases.dt[parameter].values]) 
    print('X shape:', X.shape)
    print(X[0])
    
    if scaler:
        X = scaler.fit_transform(X)
    
    return X, np.where(phases.dt[t_type].values > T, 1, 0) if T else X

def generate_test(phases, natoms=3, atoms=None, exclude=None, mode='classify'):
    """ Generates a list of phase fields not in training set. 

    Parameters
    ----------            
    phases: Phase2Vec obj
    natoms: int;          number of elements in a phase field
    atoms: list;          constituent elements of the phase fields;
                          default: None - use the same elements as in training
    exclude: list;        phase fields to exclude from the test set;
                          default: None - exclude phases listed in ICSD
    """
    if not exclude:
        exclude = pd.read_csv('DATA/icsd_phases.csv')['phases']
    if not atoms:
        atoms = phases.atoms

    test = generate(atoms, natoms, exclude)
    test = pd.DataFrame({'phases': test})
    test.to_pickle(f'Unexplored_phase_fields_{natoms}.pkl')
    test = get_phases(phases=f'Unexplored_phase_fields_{natoms}.pkl', maxlength=phases.maxlength, mode=mode)

    return test

def write_results(test, probabilities, rankings, Tc=10, to='test_results.csv'):
    """ Reads classes in test and writes to csv """

    def rr(x): return round(x,3)

    df = test.dt['phases']

    # look at the '1' class probability:
    df[f'probability_Tc>{Tc}K'] = list(map(rr, probabilities[:,1]))
    
    df['rankings'] = rankings
    normalized_df = (df['rankings']-df['rankings'].min())/(df['rankings'].max() - df['rankings'].min())
    df['norm_score'] = list(map(rr, normalized_df))

    df.sort_values([f'probability_Tc>{Tc}K'], ascending=False)
    df.to_csv(to, index=None)


if __name__ == '__main__':

    dirname = 'Magnetic_phase_fields'
    log = 'End_to_end_training_combined_data_results.txt'
    epochs = 100
    Tc = 10 # Divide phase fields by Tc threshold 

    # =============== CLASSIFICATION ================
    #phases = get_phases(phases='DATA/mpds_magnet_CurieTc.csv', mode='classify')
    phases = get_phases(phases='DATA/mpds_magnet_CurieTc.csvone_hot_phases.pkl', mode='classify')
    test = get_phases(phases='DATA/Ternary_phase_fields.pklone_hot_phases.pkl', mode='classify', maxlength=phases.maxlength)
    #test = generate_test(phases, natoms=3) 
 
    X, y  = get_dataset(phases, 'onehot', 'max Tc', T=Tc)
    X_test, _ = get_dataset(test, 'onehot')
 
    model = phases.classifier(X, y, epochs=epochs, dirname=dirname)
 
    print("Predicting classification for unexplored phase fields...")
    probabilities = model.predict(X_test)
 
    # =============== AE RANKING  ================

    # load phase fields of phase vectors if precalculated
    #phases = get_phases(phases='DATA/mpds_magnet_CurieTc.csv', mode='rank')
    #test = get_phases(phases='DATA/Ternary_phase_fields.pkl', mode='rank', maxlength=phases.maxlength)
    phases = get_phases(phases='DATA/mpds_magnet_CurieTc.csvone_hot_phases.pkl', mode='rank')
    test = get_phases(phases='DATA/Ternary_phase_fields.pklone_hot_phases.pkl', mode='rank', maxlength=phases.maxlength)
    X, _  = get_dataset(phases,'phases_vectors') 
    X_test, _ = get_dataset(test, 'phase_vectors') #, scaler=StandardScaler())
    
    model = phases.rank(X, X, epochs=epochs, dirname=dirname)

    print("Predicting chemical accessibility rankings for unexplored phase fields...")
    rankings = model.predict(X_test)
    rankings = pairwise_distances_no_broadcast(X_test, rankings)

    write_results(test, probabilities, rankings, to='Magnetic_ternaries_scores.csv')
