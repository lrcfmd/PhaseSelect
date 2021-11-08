import os
import sys
import re 
import tensorflow as tf
import numpy as np
from numba import njit
import pandas as pd
import itertools as its
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array
from pymatgen.core.composition import Composition as C

def generate(atoms, natoms, training):
    """ generates a list of sets of 'natoms' 'atoms' that are not in the 'training' list """
    new =  set(its.combinations(atoms, natoms))
    return [' '.join(map(str,i)) for i in new.difference(set(training))]

def create_callback(dirname):
    """ Create a callback that saves the model's weights """
    
    if not os.path.isdir(f"{dirname}"):
        os.mkdir(f"{dirname}")
   
    checkpoint_path = f"{dirname}"+"/weights.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             save_weights_only=True,
                                             verbose=1)
def clean_cut(phases, cut):
   """ Ensure the validation set doesn't 
       contain permutatations of the training vectors """
   from itertools import permutations

   def perms(phase):
        return [list(i) for i in permutations(phase.split())]

   print("Searching the clean validation cut...")   
   while phases[cut].split() in perms(phases[cut-1]): 
        #print('Cut at index:', cut, phases[cut])
        cut += 1 

   return cut 

def segment_classes_kfold(df, Tc=10, n_splits=5, by='phases_vectors'):
    """ Divide phase fields on subclasses by Tc threshold.
    Split data on training and validation. Kfold """
    print(f"Spliting data: training and validation {n_splits}-fold")
    
    X = np.array([i for i in df[by].values])
    t = df['max Tc'].values #np.array([i for i in df['max Tc'].values])
    y = np.where(t>Tc, 1, 0)
    
    print(f"Distribution of phases with Tc  > and < {Tc}K: {sum(y)} {len(t) - sum(y)}")

    return X, y, KFold(n_splits=n_splits, shuffle=False)

def split(X, y, df, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dftest = pd.DataFrame({'phases': df['phases'].values[test_index],
                           'max Tc': df['max Tc'].values[test_index]})  
    print('Dtest:', dftest.shape)
    return X_train, X_test, y_train, y_test, dftest

def class_weights(x):
    """ To account for data distribution imbalance in classes"""
    w0 = len(x) / len(x[x==0]) / 2
    w1 = len(x) / len(x[x!=0]) / 2
    return {0:w0, 1:w1}

def pad(s, maxlength=8):
    """ padding phase fields with zeros """
    add = maxlength - len(s)
    add = [0 for i in range(add)]
    return np.array([i for i in s] + add) if add else np.array(s)

def embedding(df, a2v, maxlength=None):
    """
    df = pd.DataFrame({'phases': list(dicT.keys()),
                    'N entries': [len(i) for i in dicT.values()],
                           'Tc': list(dicT.values()),
                     'range Tc': ranges,
                       'std Tc': std})
    """
    print ("Embedding the phase fields as vectors.")

    phases = df['phases'].values

    if maxlength is None:
        maxlength = max([len(p.split()) for p in phases]) * len(list(a2v.values())[0])
    print("Embedding length:", maxlength)
  
    def getvec(p):
        s = np.asarray([a2v[e] for e in p.split()]).flatten()
        return np.append(s, np.zeros(maxlength - len(s)))

    print ("Padding the phase fields' vectors with 0.")
    vectors = list(map(getvec, phases))

    df['phases_vectors'] = vectors
    df.to_csv('prebuilt_atoms_phases_vectors.csv')
    df.to_pickle('prebuilt_atoms_phases_vectors.pkl')
    print(df.head())
    return df, maxlength

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.

    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_fast(X, Y)


@njit
def _pairwise_distances_no_broadcast_fast(X, Y): 
    """Internal function for calculating the distance with numba. Do not use.
    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

def findatom(f):
   f = re.split("[0-9\.\-\+\=\-\!\s]+", f)
   if 'z' in f: f.remove('z')
   return ''.join(f)

def parse_phases(dbfile):
    print(f"Parsing {dbfile}...")
    df = pd.read_csv(dbfile)
    formulas = [findatom(f) for f in df.values[:,0]]
    scores = df.values[:,1]
    phases = []
    dicT = {}
    for i,f in enumerate(formulas):
        try:
           pf = C(f).as_dict()
           pf = ' '.join(sorted(pf.keys()))
        except: 
           print (f"Cannot parse {f}")
           continue

        if pf not in phases:
            phases.append(pf)
            dicT[pf] = [scores[i]]
        else:
            dicT[pf].append(scores[i])

    std, ranges = [], []
    for i in dicT.values():
        std.append(np.std(np.array(i)))
        ranges.append(np.max(i) - np.min(i))

    dt = pd.DataFrame({'phases': list(dicT.keys()), 
                       'max Tc': [max(i) for i in dicT.values()],
                      })
