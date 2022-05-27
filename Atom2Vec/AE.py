import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class AtomicEncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AtomicEncoderLayer, self).__init__()
        self.normalizer = tf.keras.layers.experimental.preprocessing.Normalization()

    def call(self, input_data, k_features):
        """ builds AE k_features """
        input_dim = len(input_data[0])
        # Normalize
        self.normalizer.adapt(input_data)
        inputs = tf.keras.Input(shape=(input_dim,)) # encoded_input
        x = normalizer(inputs)
        # encode 
        encoded = tf.keras.layers.Dense(k_features, activation='relu')(x)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
        decoded = tf.keras.layers.Reshape((input_dim,))(decoded)

        #--- extract RE-----
        #ae = tf.keras.Model(encoded_input, decoded)
        #encoder = tf.keras.Model(encoded_input, encoded)

        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, decoded)
     
        return encoded, reconstruction_loss

def train_AE(input_data, k_features):
    """ builds AE k_features """
    input_data = StandardScaler().fit_transform(input_data)
    input_dim = len(input_data[0])

    # Normalize
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(input_data)
    # Build model
    atinput = tf.keras.layers.Input(shape=(input_dim,))
    encoded_input = normalizer(atinput)
    encoded = tf.keras.layers.Dense(k_features, activation='relu')(encoded_input)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)
    ae = tf.keras.Model(atinput, decoded)
    #--- extract RE-----
    reconstruction_loss = tf.keras.losses.mean_squared_error(encoded_input, decoded)
    encoder = tf.keras.Model(atinput, encoded)
    
    early = tf.keras.callbacks.EarlyStopping(
             monitor="loss",
             min_delta=0,
             patience=10,
             verbose=0,
             mode="auto",
             baseline=None,
             restore_best_weights=False)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate=1e-3,
                  decay_steps=10000,
                  decay_rate=0.9)
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    ae.add_loss(reconstruction_loss)
    ae.compile(optimizer=optim)
    
    ae.fit(input_data, input_data,
                epochs=13,
                batch_size=16,
                callbacks=[early],
                shuffle=True)

    atomvecs = encoder.predict(input_data)

    return atomvecs, reconstruction_loss, encoder

if __name__ == '__main__':
     
    from EnvMatrix import EnvsMat
    envs_mat = EnvsMat("string.json")
    atom_vecs, _, __ = train_AE(envs_mat.envs_mat, 32)
