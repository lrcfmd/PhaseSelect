import os, sys
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as MHA
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from DB.augmentation import augment

class Endtoend(tf.keras.Model):
    def __init__(self, k_features=20):
        super(Endtoend, self).__init__()
        self.k_features = k_features

    @staticmethod
    def create_padding_mask(seq):
        tf.not_equal(seq, 0)

    def attention(self, x):
        mask = self.create_padding_mask(x)
        attention = MHA(num_heads=8,
                        key_dim=2,
                        value_dim=2)
        return attention(x, x)

    def call(self, x, atoms, attention=False):

        n = tf.shape(x)[1]  # elements in a phase field
        m = tf.shape(x)[2]  # all atoms considered 
        k = self.k_features # atomic k_features
        
        inputs = tf.keras.layers.Input(shape=(n,m,))

        # Atomic vectors extraction model
        atoms_dim = len(atoms[0])
        # Normalize
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(atoms)
        # Build model
        atinput = tf.keras.layers.Input(shape=(atoms_dim,),name='Atomic input')
        encoded_input = normalizer(atinput)
        encoded = tf.keras.layers.Dense(k, kernel_regularizer=regularizers.l1_l2(l1=0.03, l2=1e-4), activation='relu')(encoded_input) 
        encoded_drop = tf.keras.layers.Dropout(0.5)(encoded) 
        decoded = tf.keras.layers.Dense(atoms_dim, activation='sigmoid')(encoded_drop)
        decoded = tf.keras.layers.Reshape((atoms_dim,))(decoded)
     
        reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
     
        encoder = tf.keras.Model(atinput, encoded)
        encoder.add_loss(reconstruction_loss)

        embeddings = encoder(atoms)

        # phases are one-hot encodings @ atomic embeddings:
        x_ = tf.reshape(inputs, [-1, m])
        phases = tf.reshape(x_ @ embeddings, [-1, n, k])

        if attention:
            phases = self.attention(phases)

        # Dense NN
        phases = tf.reshape(phases, [-1, n*k])
        x = tf.keras.layers.Dropout(0.5)(phases)
        x = tf.keras.layers.Dense(80, kernel_regularizer=regularizers.l1_l2(l1=0.03, l2=1e-4), activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs )

        print(m.summary())

        # Write trained embeddings
        atomvec_file="Atom2Vec/atomic_embeddings.txt"
        print(f"Writing trained embeddings to {atomvec_file}")
        with open(atomvec_file, 'a') as f:
            for atom in embeddings.numpy():
                 print(' '.join(map(str,atom)), file=f)

        return m, embeddings.numpy()

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

        for n in [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/16), 4, int(seq_len/16), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]:
            x = tf.keras.layers.Dense(n, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(seq_len, activation="sigmoid")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        re = tf.keras.losses.mean_squared_error(norm_input, outputs)
        m.add_loss(re)
        print(m.summary())

        return m

if __name__ == '__main__':
  test = np.random.uniform(2,32,6960).reshape(10,8,87)
  env = np.random.uniform(2,32,3480).reshape(87,40)
  test_y = np.where(np.random.uniform(1,10,10)>5, 1, 0)

  m, embeddings = Endtoend()(test, env, attention=True)
  m.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  m.fit(test, test_y,
        batch_size=2,
        epochs=2,
        )

  print(embeddings) # identical to encoder.predict(env)
  print("Models are built")
