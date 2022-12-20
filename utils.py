import os
import sys
import re 
import tensorflow as tf
import functools
import numpy as np
from numba import njit
import pandas as pd
import itertools as its
from sklearn.utils.validation import check_array
from sklearn.model_selection import KFold
from pymatgen.core.composition import Composition as C

def augment(df, prop='max Tc'):
    dics = []
    phases = [x.split() for x in df.phases[:3]]
    onehot = df.onehot[:3]
    props = df[f'{prop}'][:3]
    for phase, onehoti, value in zip(phases, onehot, props):
        permutations = [' '.join(list(combination)) for combination in its.permutations(phase)]
        permhot = [list(combinations) for combinations in its.permutations(onehoti)]
        dics.append(pd.DataFrame({'phases': permutations,
                     'onehot': permhot,
                     f'{prop}': [value for n in range(len(permutations))]})) 
    return pd.concat(dics)

def validate(model):
    """ k-fold cross validation decorator
        input: regressor or classifier model
        returns: a list of errors for the models trained on splits"""
    @functools.wraps(model)
    def kfold(*args, **kwargs):
        n_splits = 5
        X = args[0]
        y = args[1]
        kfold = KFold(n_splits, shuffle=False)
        errors = []
        print('5-fold cross validation')
        n = 1
        for i, j in kfold.split(X):
            input_x, input_y  = X[i], y[i]
            test_x, test_y = X[j], y[j]
            _, best_val_loss = model(input_x, input_y, test_x, test_y, 
                    k_features = kwargs['k_features'], 
                    epochs = kwargs['epochs'],
                    batch_size = kwargs['batch_size'])
            errors.append(best_val_loss)
            print(f'Split 1, {model.__name__} error: {best_val_loss}')
        return errors   # Note: not a callable 'model'
    return k_fold            
    
def generate(atoms, natoms, training):
    """ generates a list of sets of 'natoms' 'atoms' that are not in the 'training' list """
    new =  set(its.combinations(atoms, natoms))
    return [' '.join(map(str,i)) for i in new.difference(set(training))]

def create_callback(dirname,batch_size):
    """ Create a callback that saves the model's weights """
    
    if not os.path.isdir(f"{dirname}"):
        os.mkdir(f"{dirname}")
   
    checkpoint_path = f"{dirname}"+"/weights.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             save_weights_only=True,
                                             verbose=1,
                                             save_freq=100*batch_size)

def clean_cut(df, validation_split=0.2, prop='max Tc'):
   """ Ensure the validation set doesn't 
       contain permutatations of the training vectors """

   print("Searching the clean validation cut...")
 
   phases = df.phases.values
   cut = int(len(phases) * (1 - validation_split))

   while set(phases[cut].split()) == set(phases[cut-1].split()): 
        print(phases[cut], phases[cut-1])
        print(len(phases), cut)
        cut += 1 

   def to_tensor(a):
       return np.array([i for i in a])

   return df.phases[:cut], to_tensor(df.onehot.values[:cut]), to_tensor(df.onehot.values[cut:]), to_tensor(df[f'{prop}'].values[:cut]), to_tensor(df[f'{prop}'].values[cut:])

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
        maxlength = max([len(p.split()) for p in phases]) * len(a2v['H'])
    print("Embedding length:", maxlength)
  
    def getvec(p):
        s = np.asarray([a2v[e] for e in p.split()]).flatten()
        return np.append(s, np.zeros(maxlength - len(s)))

    print ("Padding the phase fields' vectors with 0.")
    vectors = list(map(getvec, phases))

    df['phases_vectors'] = vectors
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

def justify_by_natoms(df, target_n=8):
    """ remove phase fields with N atoms > target_n """
    print (df.shape)
    print (df.head())
    df['N_elements'] = df.phases.apply(lambda x: len(x.split()))
    print (f'the largest phase field, N={max(df["N_elements"])}')
    df = df[df['N_elements'] <= target_n]
    df= df.drop(columns=['N_elements'])
    print(f'after removing phase fields larger than {target_n}:')
    df = df[['phases', 'max Tc']]
    print (df.shape)
    return df

if __name__=='__main__':
    file = sys.argv[1]
    #df = justify_by_natoms(pd.read_csv(file))
    #df = justify_by_natoms(pd.read_pickle(file))
    #name = file.split('.')[0]
    #df.to_pickle(f'{name}8.pkl')

#    new_phases = augment(df)
#    print(new_phases.head)
    #new_phases.to_pickle('DATA/scon_augmented.pkl')
 
