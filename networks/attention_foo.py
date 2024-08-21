import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, MultiHeadAttention, Add, Concatenate, \
    Masking, Flatten, Layer, TextVectorization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple
from utils.utils import PAD_VAL


class _TextLayer(Layer):
    def __init__(self, max_text_length: int, text_embedding_dim: int, vocab: List[str]) -> None:
        super(_TextLayer, self).__init__()
        self.max_text_length, self.text_embedding_dim, self.vocab = max_text_length, text_embedding_dim, vocab

    def build(self) -> None:
        vocab_size = len(self.vocab)
        self.text_vectorization = TextVectorization(max_tokens=vocab_size + 2, vocabulary=self.vocab,
                                                    output_sequence_length=self.max_text_length)
        self.g_text_position_embedding = Embedding(input_dim=self.max_text_length, output_dim=self.text_embedding_dim)
        self.e_text_position_embedding = Embedding(input_dim=self.max_text_length, output_dim=self.text_embedding_dim)
        self.g_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)
        self.e_text_embedding = Embedding(input_dim=vocab_size, output_dim=self.text_embedding_dim)

    def get_config(self) -> Dict:
        return {
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'vocab': self.vocab
        }

    def call(self, g_text: np.array, e_text: np.array) -> Tuple[tf.Tensor, tf.Tensor]:
        g_text_tokenized = self.text_vectorization(g_text)
        g_text_position_indices = tf.range(g_text_tokenized.shape[-1])
        g_text_embedded_pos = self.g_text_position_embedding(g_text_position_indices)
        g_text_embedded = self.g_text_embedding(g_text_tokenized)

        e_text_tokenized = self.text_vectorization(e_text)
        e_text_position_indices = tf.range(e_text_tokenized.shape[-1])
        e_text_embedded_pos = self.e_text_position_embedding(e_text_position_indices)
        e_text_embedded = self.e_text_embedding(e_text_tokenized)

        return g_text_embedded + g_text_embedded_pos, e_text_embedded + e_text_embedded_pos


class _StateLayer(Layer):
    def __init__(self, mask_value: float, state_output_dim: int, use_pos_encodings: bool, dropout_rate: float) -> None:
        super(_StateLayer, self).__init__()
        self.mask_value = mask_value
        self.state_output_dim = state_output_dim
        self.use_pos_encodings = use_pos_encodings
        self.dropout_rate = dropout_rate

    def build(self) -> None:
        self.dense = Dense(self.state_output_dim, activation='relu')
        self.position_embedding = Embedding(input_dim=self.state_output_dim, output_dim=1)
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

    def call(self, state: np.array) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(state, self.mask_value), dtype=tf.float32)
        state_masked = state * mask

        if self.use_pos_encodings:
            pos = tf.squeeze(self.position_embedding(state_masked))
            x = state_masked + pos

        else:
            x = state_masked

        x = self.dense(x)
        x = tf.reshape(x, [x.shape[0], -1, 1])
        att = self.self_attention(x, x)
        att = self.dropout(att)
        x = x + att

        return self.norm(x)


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

    def call(self, g_text_embedded: tf.Tensor, e_text_embedded: tf.Tensor, state_att: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        state_att_with_g_text = self.g_text_attention(state_att, g_text_embedded)
        state_att_with_e_text = self.e_text_attention(state_att, e_text_embedded)

        state_att_with_g_text = self.dropout_1(state_att_with_g_text)
        state_att_with_e_text = self.dropout_2(state_att_with_e_text)

        return self.norm_1(state_att + state_att_with_g_text), self.norm_2(state_att + state_att_with_e_text)


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

    def call(self, g_text: np.array, e_text: np.array, state: np.array, n_assumptions: np.array) -> tf.Tensor:
        g_text_embeddings, e_text_embeddings = self.text_layer(g_text, e_text)
        state_attention = self.state_layer(state)
        state_attention_with_g_text, state_attention_with_e_text = \
            self.cross_attention_layer(g_text_embeddings, e_text_embeddings, state_attention)
        n_assumptions_processed = self.n_assumptions_dense_1(tf.reshape(n_assumptions, [-1, 1]))
        n_assumptions_processed = self.n_assumptions_dense_2(n_assumptions_processed)

        x = tf.concat([tf.squeeze(state_attention_with_g_text), tf.squeeze(state_attention_with_e_text),
                       n_assumptions_processed], axis=1)
        x = self.final_dense_1(x)
        x = self.final_dense_2(x)
        x = self.dropout(x)
        x = self.norm(x)

        return self.output_layer(x)




