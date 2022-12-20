import os, sys
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as MHA
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd

def Atomic(input_data, k_features=20):
    """ builds AE k_features """

    #print(f"USING ATOMIC to build {k_features} atomic features")
    input_dim = len(input_data[0])
    # Normalize
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(input_data)
    # Build model
    atinput = tf.keras.layers.Input(shape=(input_dim,),name='Atomic input')
    encoded_input = normalizer(atinput)
    encoded = tf.keras.layers.Dense(k_features, activation='relu')(encoded_input)
#    encoded = tf.keras.layers.Dropout(0.1)(encoded)  #  added
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)

    ae = tf.keras.Model(atinput, decoded, name='Atomic autoencoder')
    #print(ae.summary())

    #--- extract RE-----
    reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
    ae.add_loss(reconstruction_loss)

    encoder = tf.keras.Model(atinput, encoded)
    encoder.add_loss(reconstruction_loss)
    #print(encoder.summary())

    return encoder, ae, reconstruction_loss


class Endtoend(tf.keras.Model):
    def __init__(self, k_features=40, hidden_layer=500, model_type='clf', unfix=True):
        super(Endtoend, self).__init__()
        self.k_features = k_features
        self.hidden_layer = hidden_layer
        self.model_type = model_type
        self.unfix = unfix # unfix BN layer to 'trainable'

    @staticmethod
    def create_padding_mask(seq):
        tf.not_equal(seq, 0)

    def attention(self, x):
        mask = self.create_padding_mask(x)
        attention = MHA(num_heads=8,
                        key_dim=2,
                        value_dim=2)
        return attention(x, x, return_attention_scores=True, attention_mask=mask)

    def learn_phases(self, x, atoms, encoder):
        n = tf.shape(x)[1]  # elements in a phase field: e.g. 8 max in bg
        m = tf.shape(x)[2]  # all atoms considered: e.g. 112 or 87
        k = self.k_features # atomic k_features: parameter 
        
        encoder, _, _ = Atomic(atoms, k)

        inputs = tf.keras.layers.Input(shape=(n, m,))
        x_ = tf.reshape(inputs, [-1, m])                    # phases are one-hot encodings @ atomic embeddings
        embeddings = encoder(atoms)                         # embeddings are calculated from environement matrix
        phases = tf.reshape(x_ @ embeddings, [-1, n, k])
        phases, _ = self.attention(phases)
        x__ = tf.reshape(phases, [-1, n*k])


    def call(self, x, atoms, attention=True):

        n = tf.shape(x)[1]  # elements in a phase field: e.g. 8 max in bg
        m = tf.shape(x)[2]  # all atoms considered: e.g. 112 or 87
        k = self.k_features # atomic k_features: parameter 
        hidden_layer = self.hidden_layer # parameter
        

        encoder, ae, reconstruction_loss = Atomic(atoms, k)
        #---------------- Deep NN-------------------------- 

        # input model
        inputs = tf.keras.layers.Input(shape=(n,m,))
        x_ = tf.reshape(inputs, [-1, m])                    # phases are one-hot encodings @ atomic embeddings
        embeddings = encoder(atoms)                         # embeddings are calculated from environement matrix
        phases = tf.reshape(x_ @ embeddings, [-1, n, k])

        if attention:
            phases_at, attention_scores = self.attention(phases)
            x__ = tf.reshape(phases_at, [-1, n*k])
        else:
            x__ = tf.reshape(phases, [-1, n*k])

       #input_model = tf.keras.Model(inputs, x__)
       #x0 = tf.keras.layers.BatchNormalization()(x__)
       
        x0 = tf.keras.layers.BatchNormalization()(x__)
        input_model = tf.keras.Model(inputs, x0)

        # inner model
        regularizer = regularizers.l1_l2(l1=0.01, l2=1e-3)
        inner_model = tf.keras.Sequential([
            input_model,
            tf.keras.layers.Dense(hidden_layer, kernel_regularizer=regularizer, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            # for larger data - bg, include

          # tf.keras.layers.Dense(hidden_layer, kernel_regularizer=regularizer, activation="relu"),
          # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.Dropout(0.5),
          # tf.keras.layers.Dense(hidden_layer/4, kernel_regularizer=regularizer, activation="relu"),
          # tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.Dropout(0.5)
          ])

        full_model = tf.keras.Sequential([inner_model, tf.keras.layers.Dense(1)])
        return input_model, inner_model, full_model



if __name__ == '__main__':
  test = np.random.uniform(2,32,6960).reshape(10,8,87)
  env = np.random.uniform(2,32,1740).reshape(87,20)
  m = Endtoend()(test, env, attention=True)
  test = np.random.uniform(2,32,13920).reshape(20,8,87)
  m = Endtoend()(test, env, attention=True)
  print("Models are built")
