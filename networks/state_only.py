import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Layer, Dropout
from tensorflow.keras.models import Model
from typing import Dict


# Layer that process the state vector(s)
@keras.saving.register_keras_serializable()
class StateLayer(Layer):
    def __init__(self, state_output_dim: int, use_pos_encodings: bool, dropout_rate: float) -> None:
        super(StateLayer, self).__init__()
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.dropout_rate = dropout_rate

        self.dense_1 = Dense(self.state_output_dim, activation='relu')
        self.dense_2 = Dense(self.state_output_dim, activation='relu')
        self.self_attention = MultiHeadAttention(num_heads=4, key_dim=self.state_output_dim)
        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {
            'state_output_dim': self.state_output_dim,
            'use_pos_encodings': self.use_pos_encodings,
            'dropout_rate': self.dropout_rate
        }

    def call(self, state_masked: np.array, training: bool) -> tf.Tensor:
        # Transform the masked states
        x = self.dense_1(state_masked)
        if x.shape[0] is None:
            x = tf.reshape(x, [1, self.state_output_dim])

        # If we are using positional encodings for the state (to see where each feature is located), add them to x
        if self.use_pos_encodings:
            pos = np.arange(self.state_output_dim)[:, np.newaxis]
            i = np.arange(1)[np.newaxis, :]
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(1))
            angles = pos * angle_rates
            angles[:, 0::2] = np.sin(angles[:, 0::2])
            angles[:, 1::2] = np.cos(angles[:, 1::2])
            pos_encoding = tf.cast(angles.reshape(1, -1), dtype=tf.float32)
            x = x + pos_encoding

        # Use self-attention - note that the reshaping is needed based on how TensorFlow performs multi-headed attention
        x = tf.reshape(x, [x.shape[0], -1, 1])
        att = self.self_attention(x, x)

        # Drop out, residual connection, normalize
        att = self.dropout(att, training=training)
        x = x + att
        x = tf.squeeze(self.norm_1(x))

        # Final dense layer, residual connection
        if len(x.shape) == 1:
            x = tf.reshape(x, [1, -1])
        final_dense = self.dense_2(x)
        x = x + final_dense
        x = tf.reshape(x, [x.shape[0], -1, 1])

        # Normalize and return
        return self.norm_2(x)


class StateOnly(Model):
    def __init__(self, mask_value: float, state_output_dim: int, use_pos_encodings: bool, dense_dim: int,
                 dropout_rate: float) -> None:
        super(StateOnly, self).__init__()
        self.mask_value = mask_value
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate

        self.state_layer = Dense(self.dense_dim, activation='relu')
        self.state_attention_layer = StateLayer(self.state_output_dim, self.use_pos_encodings, self.dropout_rate)
        self.dense_1 = Dense(self.dense_dim, activation='relu')
        self.dense_2 = Dense(self.dense_dim, activation='relu')
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        self.output_layer = Dense(100, activation='sigmoid')

    def get_config(self) -> Dict:
        return {
            'mask_value': self.mask_value,
            'state_output_dim': self.state_output_dim,
            'use_pos_encodings': self.use_pos_encodings,
            'dense_dim': self.dense_dim,
            'dropout_rate': self.dropout_rate
        }

    def call(self, state: np.array, training: bool = False) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(state, self.mask_value), dtype=tf.float32)
        state_masked = state * mask
        state_processed = self.state_layer(state_masked)
        state_att = self.state_attention_layer(state_masked, training=training)
        x = state_processed + tf.squeeze(state_att)

        x = x + self.dense_1(x)
        x = x + self.dense_2(x)
        x = self.dropout(x, training=training)
        x = self.norm(x)

        return self.output_layer(x)
