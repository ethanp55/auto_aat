import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization, MultiHeadAttention, \
    TextVectorization
from tensorflow.keras.models import Model
from typing import Dict, List, Tuple


# Layer that process the generator and environment/domain description(s)
@keras.saving.register_keras_serializable()
class TextLayer(Layer):
    def __init__(self, key_dim: int, max_text_length: int, text_embedding_dim: int, vocab: List[str],
                 dropout_rate: float) -> None:
        super(TextLayer, self).__init__()
        self.key_dim = key_dim
        self.max_text_length, self.text_embedding_dim, self.vocab = max_text_length, text_embedding_dim, vocab
        self.dropout_rate = dropout_rate

        vocab_size = len(self.vocab)
        self.text_vectorization = TextVectorization(max_tokens=vocab_size + 2, vocabulary=self.vocab,
                                                    output_sequence_length=self.max_text_length)
        self.g_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)
        self.g_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.e_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)
        self.e_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)
        self.dropout_3 = Dropout(self.dropout_rate)
        self.dropout_4 = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {
            'key_dim': self.key_dim,
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'vocab':  keras.saving.serialize_keras_object(self.vocab),
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        vocab_config = config.pop('vocab')
        vocab = keras.saving.deserialize_keras_object(vocab_config)
        config['vocab'] = vocab

        return cls(**config)

    def call(self, data_tup: Tuple[np.array, np.array], training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        g_text, e_text = data_tup

        # Calculate positional encodings
        pos = np.arange(self.max_text_length)[:, np.newaxis]
        i = np.arange(self.text_embedding_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.text_embedding_dim))
        angles = pos * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = tf.cast(angles[np.newaxis, ...], dtype=tf.float32)

        # Tokenize and embed the generator description(s)
        g_text_tokenized = self.text_vectorization(g_text)
        g_text_embedded = self.g_text_embedding(g_text_tokenized)
        g_text_embedded = self.dropout_1(g_text_embedded + pos_encoding, training=training)
        g_text_att = self.g_text_attention(g_text_embedded, g_text_embedded)

        # Tokenize and embed the environment description(s)
        e_text_tokenized = self.text_vectorization(e_text)
        e_text_embedded = self.e_text_embedding(e_text_tokenized)
        e_text_embedded = self.dropout_2(e_text_embedded + pos_encoding, training=training)
        e_text_att = self.e_text_attention(e_text_embedded, e_text_embedded)

        return self.dropout_3(g_text_att, training=training), self.dropout_4(e_text_att, training=training)


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


# Layer that performs cross attention with process state vector(s) and the generator and environment description vectors
@keras.saving.register_keras_serializable()
class CrossAttentionLayer(Layer):
    def __init__(self, key_dim: int, dropout_rate: float) -> None:
        super(CrossAttentionLayer, self).__init__()
        self.key_dim, self.dropout_rate = key_dim, dropout_rate

        self.g_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.e_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.norm = LayerNormalization()
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {'key_dim': self.key_dim, 'dropout_rate': self.dropout_rate}

    def call(self, data_tup: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool) -> tf.Tensor:
        g_text_embedded, e_text_embedded, state_att = data_tup

        # Attention between the processed state vector(s) and the processed generator description vector(s)
        state_att_with_g_text = self.g_text_attention(state_att, g_text_embedded)
        state_att_with_e_text = self.e_text_attention(state_att, e_text_embedded)

        # Pass through dropout
        state_att_with_g_text = self.dropout_1(state_att_with_g_text, training=training)
        state_att_with_e_text = self.dropout_2(state_att_with_e_text, training=training)

        # Residual connections, normalize, and return
        return self.norm(state_att + state_att_with_g_text + state_att_with_e_text)


class ATTention(Model):
    def __init__(self, key_dim: int, max_text_length: int, text_embedding_dim: int, vocab: List[str],
                 mask_value: float, state_output_dim: int, use_pos_encodings: bool, dense_dim: int,
                 dropout_rate: float) -> None:
        super(ATTention, self).__init__()
        self.key_dim = key_dim
        self.max_text_length = max_text_length
        self.text_embedding_dim = text_embedding_dim
        self.vocab = vocab
        self.mask_value = mask_value
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate

        self.text_layer = TextLayer(self.key_dim, self.max_text_length, self.text_embedding_dim, self.vocab,
                                    self.dropout_rate)
        self.cross_attention_layer = CrossAttentionLayer(self.key_dim, self.dropout_rate)
        self.state_dense = Dense(self.dense_dim, activation='relu')
        self.state_attention_layer = StateLayer(self.state_output_dim, self.use_pos_encodings, self.dropout_rate)
        self.dense_1 = Dense(self.dense_dim, activation='relu')
        self.dense_2 = Dense(self.dense_dim, activation='relu')
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        self.output_layer = Dense(100, activation='sigmoid')

    def get_config(self) -> Dict:
        return {
            'key_dim': self.key_dim,
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'vocab':  keras.saving.serialize_keras_object(self.vocab),
            'mask_value': self.mask_value,
            'state_output_dim': self.state_output_dim,
            'use_pos_encodings': self.use_pos_encodings,
            'dense_dim': self.dense_dim,
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        vocab_config = config.pop('vocab')
        vocab = keras.saving.deserialize_keras_object(vocab_config)
        config['vocab'] = vocab

        return cls(**config)

    def call(self, data_tup: Tuple[np.array, np.array, np.array], training: bool = False) -> tf.Tensor:
        g_text, e_text, state = data_tup
        g_text_processed, e_text_processed = self.text_layer((g_text, e_text), training=training)

        mask = tf.cast(tf.not_equal(state, self.mask_value), dtype=tf.float32)
        state_masked = state * mask
        state_processed = self.state_dense(state_masked)
        state_att = self.state_attention_layer(state_masked, training=training)
        state_attention_with_text = self.cross_attention_layer((g_text_processed, e_text_processed, state_att),
                                                               training=training)
        x = state_processed + tf.squeeze(state_att) + tf.squeeze(state_attention_with_text)

        x = x + self.dense_1(x)
        x = x + self.dense_2(x)
        x = self.dropout(x, training=training)
        x = self.norm(x)

        return self.output_layer(x)
