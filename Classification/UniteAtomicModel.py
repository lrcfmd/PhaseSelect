import os, sys
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as MHA
import numpy as np
import pandas as pd
from DB.augmentation import augment

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
        k_features = k
        atoms_dim = len(atoms[0])
        
        inputs_phases = tf.keras.layers.Input(shape=(n,m,), name='Phases input')
        atinput = tf.keras.layers.Input(shape=(atoms_dim,),name='Atomic input')

        # Normalize
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(atoms)
        # Build atomic model
        encoded_input = normalizer(atinput)
        encoded = tf.keras.layers.Dense(k_features, activation='relu', name='EncodiAtoms')(encoded_input)
        decoded = tf.keras.layers.Dense(atoms_dim, activation='sigmoid', name='DecodiAtoms')(encoded)
        decoded = tf.keras.layers.Reshape((atoms_dim,))(decoded)
     
        reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
     
#        encoder = tf.keras.Model(atinput, encoded, name='Atomic Encoder')
#        encoder.add_loss(reconstruction_loss)
#        embeddings = encoder(atoms)
        embeddings = encoded

        # phases are one-hot encodings @ atomic embeddings:
        x_ = tf.keras.layers.Reshape((n*m,))(inputs_phases)
        phases = tf.keras.layers.Multiply()(x_, embeddings)
        phases = tf.keras.layers.Reshape((n, k))(phases)
        #x_ = tf.reshape(inputs_phases, [-1, m])
        #phases = tf.reshape(x_ @ embeddings, [-1, n, k])

        #phases = tf.reshape(x_ @ embeddings, [-1, n* k]) # padding?

        if attention:
            phases = self.attention(phases)

        # Build phase classification model
        phases = tf.keras.layers.Reshape((-1, n*k))(phases)
        #phases = tf.reshape(phases, [-1, n*k])
        x = tf.keras.layers.Dropout(0.1)(phases)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax", name='PhaseClass_Tc')(x)

        model = tf.keras.Model(inputs=[inputs_phases,atinput], outputs=[outputs,encoded])

        print(model.summary())

        return model

if __name__ == '__main__':
  test = np.random.uniform(2,32,6960).reshape(10,8,87)
  env = np.random.uniform(2,32,1740).reshape(87,20)
  m = Endtoend()(test, env, attention=True)
  tf.keras.utils.plot_model(m, "united_atomic_model.png", show_shapes=True)
  print("Models are built")
