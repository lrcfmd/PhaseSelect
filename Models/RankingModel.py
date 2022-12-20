import os, sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention as MHA
import numpy as np
import pandas as pd
from DB.augmentation import augment

def Atomic(input_data, k_features=20):
    """ builds AE k_features """
    input_dim = len(input_data[0])
    # Normalize
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(input_data)
    # Build model
    atinput = tf.keras.layers.Input(shape=(input_dim,),name='Atomic input')
    encoded_input = normalizer(atinput)
    encoded = tf.keras.layers.Dense(k_features, activation='relu')(encoded_input)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)

    ae = tf.keras.Model(atinput, decoded, name='Atomic autoencoder')
    print(ae.summary())
    
    reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)

    encoder = tf.keras.Model(atinput, encoded)
    encoder.add_loss(reconstruction_loss)
    print(encoder.summary())

    return encoder

class Endtoend(tf.keras.Model):
    def __init__(self):
        super(Endtoend, self).__init__()

    @staticmethod
    def create_padding_mask(seq):  # TODO - FIX
        #seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        #return seq[:, tf.newaxis, tf.newaxis, :]
        return tf.keras.layers.Masking()(seq)

    def attention(self, x):
        mask = self.create_padding_mask(x)
        attention = MHA(num_heads=2,
                        key_dim=2,
                        value_dim=2)
        return attention(x, x)

    def call(self, x, atoms, attention=False):
        n = tf.shape(x)[1] # elements in a phase field
        m = tf.shape(x)[2] # all atoms considered 
        k = 20             # atomic k_features
        
        inputs = tf.keras.layers.Input(shape=(n,m,), name='Phase input')

        encoder = Atomic(atoms, k_features=k)
        embeddings = encoder(atoms)

        # phases are one-hot encodings @ atomic embeddings:
        x_ = tf.reshape(inputs, [-1, m])
        phases = tf.reshape(x_ @ embeddings, [-1, n, k])
        #phases = tf.reshape(x_ @ embeddings, [-1, n* k]) # padding?

        if attention:
            phases = self.attention(phases)

        # Dense NN
        phases_in = tf.reshape(phases, [-1, n*k])
        x = tf.keras.layers.Dropout(0.1)(phases_in)
        mid = tf.keras.layers.Dense(80, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(mid)
        mid = tf.keras.layers.Dense(20, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(mid)
        mid = tf.keras.layers.Dense(n*k, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(mid)
        phases_o = tf.keras.layers.Dense(n*k, activation="sigmoid")(x)
        phases_out = tf.reshape(phases_o, [-1, n,k])
        cosine = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        rankings = cosine(phases_in, phases_o)

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        print('Rankings:', tf.shape(rankings))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        print('phases_in:', tf.shape(phases_in))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        print('phases_out:', tf.shape(phases_out))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        m = tf.keras.Model(inputs=inputs, outputs=phases_out)
        #re_loss = tf.keras.losses.binary_crossentropy(phases_in, phases_o)
        re_loss = tf.keras.losses.mean_squared_error(phases_in, phases_o)
        m.add_loss(re_loss)

        print(m.summary())
        return m, rankings

if __name__ == '__main__':
  test = np.random.uniform(2,32,6960).reshape(10,8,87)
  env = np.random.uniform(2,32,1740).reshape(87,20)
  m, r = Endtoend()(test, env, attention=True)
  tf.keras.utils.plot_model(m, "ranking_model.png", show_shapes=True)
  print("Models are built")
