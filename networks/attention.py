import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, MultiHeadAttention, Add, Concatenate, \
    Masking, Flatten, Layer, TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.utils import PAD_VAL


class CustomMasking(Layer):
    def __init__(self, mask_value, **kwargs):
        super(CustomMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, self.mask_value), dtype=tf.float32)

        return inputs * mask


def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    return pos * angle_rates


def _positional_encoding(position, d_model):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices in the array
    p_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(p_encoding, dtype=tf.float32)


# Masking during training
def _custom_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, PAD_VAL)
    loss = tf.keras.losses.MSE(y_true * tf.cast(mask, tf.float32), y_pred * tf.cast(mask, tf.float32))
    loss = tf.reduce_mean(loss)

    return loss


# Text processing with embedding and positional encoding
vocab_size = 10000
embedding_dim = 128
max_text_length = 100
text_input_1 = Input(shape=(1,), dtype=tf.string)
text_input_2 = Input(shape=(1,), dtype=tf.string)
text_vectorizer = TextVectorization(max_tokens=vocab_size, ragged=False, output_sequence_length=max_text_length)
text_vectorized_1 = text_vectorizer(text_input_1)
text_vectorized_2 = text_vectorizer(text_input_2)
text_embedding_1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_vectorized_1)
text_embedding_2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_vectorized_2)
pos_encoding = _positional_encoding(max_text_length, embedding_dim)
text_embedding_1 += pos_encoding
text_embedding_2 += pos_encoding

# Process the environment state with self-attention and positional encodings
state_embedding_dim = 128
state_input = Input(shape=(1, 300), dtype=tf.float32)
masked_input = CustomMasking(mask_value=PAD_VAL)(state_input)
state_embedding = Dense(state_embedding_dim)(masked_input)
# pos_encoding_state = _positional_encoding(state_input.shape[1], state_embedding_dim)
# pos_encoding_state = _positional_encoding(state_embedding_dim, state_embedding_dim)
# state_embedding += pos_encoding_state
self_attention_state = MultiHeadAttention(num_heads=4, key_dim=state_embedding_dim)(state_embedding, state_embedding)
self_attention_state = LayerNormalization()(Add()([state_embedding, self_attention_state]))

# Attention between processed environment state and text descriptions
attention_text_state_1 = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(self_attention_state, text_embedding_1)
attention_text_state_2 = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(self_attention_state, text_embedding_2)
combined_attention_1 = LayerNormalization()(Add()([self_attention_state, attention_text_state_1]))
combined_attention_2 = LayerNormalization()(Add()([self_attention_state, attention_text_state_2]))

# Process the assumption count input
assumptions_count_input = Input(shape=(1, 1), dtype=tf.float32)
assumptions_count_dense = Dense(state_embedding_dim)(assumptions_count_input)

# Combine processed inputs
combined_input = Concatenate()([attention_text_state_1, attention_text_state_2, assumptions_count_dense])

# Feed-forward network
x = Flatten()(combined_input)
x = Dense(256, activation='relu')(x)
x = LayerNormalization()(x)
x = Dense(128, activation='relu')(x)
x = LayerNormalization()(x)

# Output layer
output = Dense(100, activation='sigmoid')(x)

# Define the model
model = Model(inputs=[text_input_1, text_input_2, assumptions_count_input, state_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=_custom_loss)

# Summary of the model
print(model.summary())
