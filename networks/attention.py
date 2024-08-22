import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, Layer, TextVectorization, \
    Dropout
from tensorflow.keras.models import Model
from typing import Dict, List, Tuple


# Layer that process the generator and environment/domain description(s)
class _TextLayer(Layer):
    def __init__(self, max_text_length: int, text_embedding_dim: int, vocab: List[str]) -> None:
        super(_TextLayer, self).__init__()
        self.max_text_length, self.text_embedding_dim, self.vocab = max_text_length, text_embedding_dim, vocab

    def build(self) -> None:
        vocab_size = len(self.vocab)
        self.text_vectorization = TextVectorization(max_tokens=vocab_size + 2, vocabulary=self.vocab,
                                                    output_sequence_length=self.max_text_length)
        self.g_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)
        self.e_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)

    def get_config(self) -> Dict:
        return {
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'vocab': self.vocab
        }

    def call(self, g_text: np.array, e_text: np.array) -> Tuple[tf.Tensor, tf.Tensor]:
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

        # Tokenize and embed the environment/domain description(s)
        e_text_tokenized = self.text_vectorization(e_text)
        e_text_embedded = self.e_text_embedding(e_text_tokenized)

        # Add the embedded vectors with the positional encodings, return
        return g_text_embedded + pos_encoding, e_text_embedded + pos_encoding


# Layer that process the state vector(s)
class _StateLayer(Layer):
    def __init__(self, mask_value: float, state_output_dim: int, use_pos_encodings: bool, dropout_rate: float) -> None:
        super(_StateLayer, self).__init__()
        self.mask_value = mask_value
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.dropout_rate = dropout_rate

    def build(self) -> None:
        self.dense = Dense(self.state_output_dim, activation='relu')
        self.self_attention = MultiHeadAttention(num_heads=4, key_dim=self.state_output_dim)
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {
            'mask_value': self.mask_value,
            'state_output_dim': self.state_output_dim,
            'use_pos_encodings': self.use_pos_encodings,
            'dropout_rate': self.dropout_rate
        }

    def call(self, state: np.array, training: bool) -> tf.Tensor:
        # States can have different dimensions (up to 300, a value we chose) - mask any padded state values
        mask = tf.cast(tf.not_equal(state, self.mask_value), dtype=tf.float32)
        state_masked = state * mask

        # Transform the masked states
        x = self.dense(state_masked)

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

        # Drop out and residual connection
        att = self.dropout(att, training=training)
        x = x + att

        # Normalize and return
        return self.norm(x)


# Layer that performs cross attention with process state vector(s) and the generator and environment description vectors
class _CrossAttentionLayer(Layer):
    def __init__(self, key_dim: int, dropout_rate: float) -> None:
        super(_CrossAttentionLayer, self).__init__()
        self.key_dim, self.dropout_rate = key_dim, dropout_rate

    def build(self) -> None:
        self.g_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.e_text_attention = MultiHeadAttention(num_heads=4, key_dim=self.key_dim)
        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)

    def get_config(self) -> Dict:
        return {'key_dim': self.key_dim, 'dropout_rate': self.dropout_rate}

    def call(self, g_text_embedded: tf.Tensor, e_text_embedded: tf.Tensor, state_att: tf.Tensor, training: bool) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        # Attention between the processed state vector(s) and the processed generator description vector(s)
        state_att_with_g_text = self.g_text_attention(state_att, g_text_embedded)

        # Attention between the processed state vector(s) and the processed environment/domain description vector(s)
        state_att_with_e_text = self.e_text_attention(state_att, e_text_embedded)

        # Pass both through dropout
        state_att_with_g_text = self.dropout_1(state_att_with_g_text, training=training)
        state_att_with_e_text = self.dropout_2(state_att_with_e_text, training=training)

        # Residual connections, normalize, and return
        return self.norm_1(state_att + state_att_with_g_text), self.norm_2(state_att + state_att_with_e_text)


# The full AATention model that receives as input:
#   - Text descriptions of generators we are making predictions for
#   - Text descriptions of the environments/domains in which the generators operate
#   - State vectors that represent the environment/domain state in which we want to make assumption predictions
#   - Integer vectors that represent how many assumptions the generator relies on (min of 1, max of 100)
# The model then outputs a 100 dimensional vector of values between 0 and 1, representing the estimated assumption
# values for the generator(s)
class AATention(Model):
    def __init__(self, max_text_length: int, text_embedding_dim: int, vocab: List[str], mask_value: float,
                 state_output_dim: int, use_pos_encodings: bool, key_dim: int, final_dense_1_dim: int,
                 final_dense_2_dim: int, dropout_rate: float) -> None:
        super(AATention, self).__init__()
        self.max_text_length = max_text_length
        self.text_embedding_dim = text_embedding_dim
        self.vocab = vocab
        self.mask_value = mask_value
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.key_dim = key_dim
        self.final_dense_1_dim = final_dense_1_dim
        self.final_dense_2_dim = final_dense_2_dim
        self.dropout_rate = dropout_rate

    def build(self) -> None:
        self.text_layer = _TextLayer(self.max_text_length, self.text_embedding_dim, self.vocab)
        self.state_layer = _StateLayer(self.mask_value, self.state_output_dim, self.use_pos_encodings,
                                       self.dropout_rate)
        self.cross_attention_layer = _CrossAttentionLayer(self.key_dim, self.dropout_rate)
        self.n_assumptions_dense_1 = Dense(int(self.state_output_dim / 2), activation='relu')
        self.n_assumptions_dense_2 = Dense(self.state_output_dim, activation='relu')
        self.final_dense_1 = Dense(self.final_dense_1_dim, activation='relu')
        self.final_dense_2 = Dense(self.final_dense_2_dim, activation='relu')
        self.norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        self.output_layer = Dense(100, activation='sigmoid')

    def get_config(self) -> Dict:
        return {
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'vocab': self.vocab,
            'mask_value': self.mask_value,
            'state_output_dim': self.state_output_dim,
            'use_pos_encodings': self.use_pos_encodings,
            'key_dim': self.key_dim,
            'final_dense_1_dim': self.final_dense_1_dim,
            'final_dense_2_dim': self.final_dense_2_dim,
            'dropout_rate': self.dropout_rate
        }

    def call(self, g_text: np.array, e_text: np.array, state: np.array, n_assumptions: np.array,
             training: bool = False) -> tf.Tensor:
        # Process the generator and environment/domain descriptions
        g_text_embeddings, e_text_embeddings = self.text_layer(g_text, e_text)

        # Process the state vector(s)
        state_attention = self.state_layer(state, training=training)

        # Perform cross attention
        state_attention_with_g_text, state_attention_with_e_text = \
            self.cross_attention_layer(g_text_embeddings, e_text_embeddings, state_attention, training=training)

        # Transform the number of assumptions
        n_assumptions_processed = self.n_assumptions_dense_1(tf.reshape(n_assumptions, [-1, 1]))
        n_assumptions_processed = self.n_assumptions_dense_2(n_assumptions_processed)

        # Concatenate the processed vectors together
        x = tf.concat([tf.squeeze(state_attention_with_g_text), tf.squeeze(state_attention_with_e_text),
                       n_assumptions_processed], axis=1)

        # Pass through final dense layers
        x = self.final_dense_1(x)
        x = self.final_dense_2(x)

        # Dropout and norm
        x = self.dropout(x, training=training)
        x = self.norm(x)

        # Pass through the output layer, which uses a sigmoid activation function (to force values between 0 and 1)
        return self.output_layer(x)




