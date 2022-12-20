import os, sys
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as MHA
import numpy as np
import pandas as pd
#import progressbar
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

class RankingAE_attention(tf.keras.Model):
    def __init__(self, k_features):
        super(RankingAE_attention, self).__init__()
        self.k_features = k_features

    def attention(self, x, return_attention_scores=True):
        attented = MHA(num_heads=2,
                        key_dim=2,
                        value_dim=2)
        return attented(x, x, return_attention_scores=return_attention_scores)

    def call(self, x):

        n = tf.shape(x)[1] # elements in a phase field
        m = tf.shape(x)[2] # all atoms considered
        k = self.k_features # atomic k_features

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(n,m,))
        x = normalizer(inputs)
        flat_x = tf.reshape(x, [-1, n*k])
        
        x, attention_scores = self.attention(x)
        x = tf.reshape(x, [-1, n*k])

        for net in [20, int(n*k)]: #[160, 80, 40, 20, 10, 20, 40, 80, 160]:
            x = tf.keras.layers.Dense(net, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(n*k, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        re = tf.keras.losses.mean_squared_error(flat_x, outputs)
        model.add_loss(re)
        print(model.summary())

        # return attention scores
        
        get_attention = tf.keras.Model(inputs, attention_scores)

        return model, get_attention

class RankingAE(tf.keras.Model):
    def __init__(self):
        super(RankingAE, self).__init__()

    def call(self, x):
        seq_len = tf.shape(x)[1]

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(seq_len,))
        norm_input = normalizer(inputs)
        x = tf.keras.layers.Dropout(0.1)(norm_input)

       # for n in [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/16), 4, int(seq_len/16), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]:
        for n in [seq_len, 120, seq_len]: 
            x = tf.keras.layers.Dense(n, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(seq_len, activation="sigmoid")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        re = tf.keras.losses.mean_squared_error(norm_input, outputs)
        m.add_loss(re)
        print(m.summary())

        return m

class Regressor(tf.keras.Model):
    def __init__(self):
        super(Regressor, self).__init__()

    def call(self, x):
        seq = tf.shape(x)[1]

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(seq,),name='My test input shape')

        x = normalizer(inputs)

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation="softmax")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(m.summary())

        return m

class NN_binary(tf.keras.Model):
    def __init__(self):
        super(NN_binary, self).__init__()

    def call(self, x):
        x = tf.convert_to_tensor(x)
        #print("On call, received input shape:", x.shape)
        #print("On call, received dynamic shape:", tf.shape(x))

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(160,),name='My test input shape')
        inputs.set_shape((None,160))

        x = normalizer(inputs)

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(m.summary())  
 
        return m

class NN_3_softmax(tf.keras.Model):
    def __init__(self):
        super(NN_3_softmax, self).__init__()

    def call(self, x):
        seq_len = tf.shape(x)[1]

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(seq_len,))
        x = normalizer(inputs)

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(m.summary())

        return m

def create_callback(dirname):
    """ Create a callback that saves the model's weights """
  
    for dirname in [dirname, f'{dirname}/weights']:
        if not os.path.isdir(f"{dirname}"):
            os.mkdir(f"{dirname}")
   
    # each epoch 
    #checkpoint_path = f"{dirname}"+"/weights.{epoch:02d}-{val_loss:.3f}.hdf5"
    checkpoint_path = f"{dirname}"+"/weights.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             save_weights_only=True,
                                             verbose=0)

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

def segment_3_classes(df, T1=1, T2=10, n_splits=5):
    """ Divide phase fields in 3 groups """
    print("Dividing data into 3 groups")

    X = np.array([i for i in df['phase_vectors'].values])
    t = np.array([i for i in df['max Tc'].values])
    y = np.where((T1<t) & (t<=T2), 1, 0) + np.where(t>T2, 2, 0)

    print(f"Distribution of phases with 0 < {T1} <  {T2} < High Tc: {len(y[y==0])} {len(y[y==1])} {len(y[y==2])}")

    return X, y, KFold(n_splits=n_splits, shuffle=False) 

def segment_regress_kfold(df, n_splits=5):
    print(f"Spliting data: training and validation {n_splits}-fold")
    X = np.array([i for i in df['phase_vectors'].values])
    y = np.array([i for i in df['max Tc'].values]) #.reshape(-1,1)
    return X, y, KFold(n_splits=n_splits, shuffle=False)

def segment_classes_kfold(df, Tc=10, n_splits=5, by='phases_vectors', prop='max Tc'):
    """ Divide phase fields on subclasses by Tc threshold.
    Split data on training and validation. Kfold """
    print(f"Spliting data: training and validation {n_splits}-fold")
    
    X = np.array([i for i in df[by].values])
    t = df[prop].values #np.array([i for i in df['max Tc'].values])
    y = np.where(t>Tc, 1, 0)
    
    print(f"Distribution of phases with Tc  > and < {Tc}K: {sum(y)} {len(t) - sum(y)}")

    return X, y, KFold(n_splits=n_splits, shuffle=False)

def split(X, y, df, train_index, test_index, prop='max Tc'):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dftest = pd.DataFrame({'phases': df['phases'].values[test_index],
                           prop: df[prop].values[test_index]})  
    print('Dtest:', dftest.shape)
    return X_train, X_test, y_train, y_test, dftest

def class_weights(x):
    """ To account for data distribution imbalance in classes"""
    w0 = len(x) / len(x[x==0]) / 2
    w1 = len(x) / len(x[x!=0]) / 2
    return {0:w0, 1:w1}

def three_class_weights(x):
    """ To account for data distribution imbalance in classes"""
    w0 = len(x) / len(x[x==0]) / 3
    w1 = len(x) / len(x[x==1]) / 3
    w2 = len(x) / len(x[x==2]) / 3
    return {0:w0, 1:w1, 2:w2}

def RandomForest(X, y, X_test, dftest, dirname):
    """ RandomForest """
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    y_test = clf.predict(X_test)
    print("Saving predictions to file ...")
    y_pred = np.where(y_test[1] >= 0.5, 1, 0)
    #for i,y in enumerate(y_test):
    #    print(dftest.values[i,:], y, file=open(f'{dirname}/2_class_prediction.doc','a'))

    return dftest.assign(prediction=y_pred)

def pad(s, maxlength=8):
    """ padding phase fields with zeros """
    add = maxlength - len(s)
    add = [0 for i in range(add)]
    return np.array([i for i in s] + add) if add else np.array(s)

def embedding_attention(df, a2v, maxlength=None):
    print ("Embedding the phase fields as vectors.")

    phases = df['phases'].values

    if maxlength is None:
        maxlength = max([len(p.split()) for p in phases])
    print("Embedding length:", maxlength)

    def padd(p):
        s = np.asarray([e for e in p.split()])
        return np.append(s, np.zeros(maxlength - len(s)))

    print ("Padding the phase fields' vectors with 0.")
    vectors = list(map(padd, phases))

    def getvec(p):
        return np.asarray([a2v[e] if e != '0.0' else np.zeros(len(a2v['Li'])) for e in p])

    print ("Embedding phase fields' vectors...")
    vectors = list(map(getvec, vectors))
    df['phases_vectors'] = vectors
    print('Shape of phase vectors:', vectors[0].shape)

    return df, maxlength


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
    lengths = np.array([len(p.split()) for p in phases])
    longest = max(lengths)

    if maxlength is None:
        maxlength = longest * len(list(a2v.values())[0])
    print("Embedding length:", maxlength)
  
    def getvec(p):
        s = np.asarray([a2v[e] for e in p.split()]).flatten()
        return s 

    vectors = list(map(getvec, phases))

    def pad(s):
        return np.append(s, np.zeros(maxlength - len(s)))
    
    if phases[lengths < longest].size > 0: 
        print ("Padding the phase fields' vectors with 0.")
        vectors = list(map(pad, vectors))

    df['phases_vectors'] = vectors
    df.to_csv('prebuilt_atoms_phases_vectors.csv')
    df.to_pickle('prebuilt_atoms_phases_vectors.pkl')
    print(df.head())
    return df, maxlength


if __name__ == '__main__':
   prediction = np.where(y_test[:,1] > 0.5, 1, 0)
   for i,y in enumerate(y_test):
       print( y, file=open('y_test.txt', 'a'))
