import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim, latent_dim=16):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(inputs)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    outputs = layers.Dense(input_dim if input_dim > 1 else 1)(decoded)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

class MSEHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.mse_list = []
    def on_epoch_end(self, epoch, logs=None):
        self.mse_list.append(logs.get('loss'))
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: MSE = {logs.get('loss'):.6f}")
