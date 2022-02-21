import os, sys
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as MHA
from tensorflow.keras import regularizers
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
    encoded = tf.keras.layers.Dropout(0.5)(encoded)  #  added
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)

    ae = tf.keras.Model(atinput, decoded, name='Atomic autoencoder')
    print(ae.summary())

    #--- extract RE-----
    reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
    ae.add_loss(reconstruction_loss)

    encoder = tf.keras.Model(atinput, encoded)
    encoder.add_loss(reconstruction_loss)
    print(encoder.summary())

    return encoder, ae, reconstruction_loss


class Endtoend(tf.keras.Model):
    def __init__(self, k_features=40):
        super(Endtoend, self).__init__()
        self.k_features = k_features

    @staticmethod
    def create_padding_mask(seq):  # TODO - FIX
        #embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=8, input_length=8, mask_zero=True) 
        #return tf.keras.layers.Masking(mask_value=0)(seq)
        #return embedding(seq)
        tf.not_equal(seq, 0)

    def attention(self, x):
        mask = self.create_padding_mask(x)
        attention = MHA(num_heads=8,
                        key_dim=2,
                        value_dim=2)
        return attention(x, x, return_attention_scores=True, attention_mask=mask)

    def call(self, x, atoms, attention=True):

        n = tf.shape(x)[1]  # elements in a phase field
        m = tf.shape(x)[2]  # all atoms considered 
        k = self.k_features # atomic k_features
        
        inputs = tf.keras.layers.Input(shape=(n,m,))

#----------------------------------------  put Atomic here instead of calling a function
        atoms_dim = len(atoms[0])
        regularizer = regularizers.l1_l2(l1=0.03, l2=1e-4)
        #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=1e-4),
        # Normalize
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(atoms)
        # Build model
        atinput = tf.keras.layers.Input(shape=(atoms_dim,),name='Atomic input')
        encoded_input = normalizer(atinput)
        encoded = tf.keras.layers.Dense(k, kernel_regularizer=regularizer, activation='relu')(encoded_input)
        encoded_drop = tf.keras.layers.Dropout(0.5)(encoded)  #  added
        decoded = tf.keras.layers.Dense(atoms_dim, activation='sigmoid')(encoded_drop)
        decoded = tf.keras.layers.Reshape((atoms_dim,))(decoded)
     
        reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
     
        encoder = tf.keras.Model(atinput, encoded)
        encoder.add_loss(reconstruction_loss)
#----------------------------------------  
#
#       encoder, ae, re = Atomic(atoms, k_features=k)
        embeddings = encoder(atoms)
#
#----------------------------------------  

        # phases are one-hot encodings @ atomic embeddings:
        x_ = tf.reshape(inputs, [-1, m])
        phases = tf.reshape(x_ @ embeddings, [-1, n, k])

        if attention:
            phases, attention_scores = self.attention(phases)

        # Dense NN
        phases = tf.reshape(phases, [-1, n*k])
        x = tf.keras.layers.Dropout(0.5)(phases)
        x = tf.keras.layers.Dense(80, kernel_regularizer=regularizer, activation="relu")(x) 
       #x = tf.keras.layers.Dense(80, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
       #x = tf.keras.layers.Dense(80, activation="relu")(x)
       #x = tf.keras.layers.Dropout(0.5)(x)
       # x = tf.keras.layers.Dense(20, kernel_regularizer=regularizers.l1_l2(l1=0.03, l2=1e-4), activation="relu")(x) #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=1e-4),
       # x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs )

        print(m.summary())

        # return attention scores

        get_attention = tf.keras.Model(inputs, attention_scores)

        return m, get_attention 

if __name__ == '__main__':
  test = np.random.uniform(2,32,6960).reshape(10,8,87)
  env = np.random.uniform(2,32,1740).reshape(87,20)
  m = Endtoend()(test, env, attention=True)
  print("Models are built")
