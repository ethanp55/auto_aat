# from networks.attention import model
from networks.attention_foo import AATention
import nltk
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from train.custom_min_max_scaler import CustomMinMaxScaler
from train.data_handler import DataHandler
from utils.utils import PAD_VAL
nltk.download('words')
from nltk.corpus import words

# TODO:
#   - Thursday: positional encodings
#   - Thursday: start training and debugging
#   - Friday: finish training and debugging, start analysis (if time)

# Parameters
TRAIN_PERCENTAGE = 0.7  # How much data to use for training
VALIDATION_PERCENTAGE = 0.3  # How much of the training data to use for validation
NETWORK_NAME = 'attention'
N_EPOCHS = 10
EARLY_STOP = int(N_EPOCHS * 0.2)
BATCH_SIZE = 16

# Grab the data
states, g_text, e_text, n_assumptions, a_vectors = \
    DataHandler.extract_domain_data(domain='jhg', augment_states=False)

# Calculate how many samples to use for training and validation (and, as a result, testing)
n_samples = states.shape[0]
n_train = int(n_samples * TRAIN_PERCENTAGE)
n_val = int(n_train * VALIDATION_PERCENTAGE)

# Randomly shuffle the data indices
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Grab the training, validation, and test indices - use assertions to make sure we extracted everything properly
train_indices = indices[:n_train - n_val]
val_indices = indices[n_train - n_val:n_train]
test_indices = indices[n_train:]
assert len(val_indices) == n_val
assert len(train_indices) + len(val_indices) == n_train
assert len(train_indices) + len(val_indices) + len(test_indices) == n_samples

# Grab the training, validation, and test sets - use assertions to make sure we extracted everything properly
states_train, g_text_train, e_text_train, n_assumptions_train, y_train = \
    states[train_indices, :], g_text[train_indices, ], e_text[train_indices, ], n_assumptions[train_indices, ], a_vectors[train_indices, :]
states_val, g_text_val, e_text_val, n_assumptions_val, y_val = \
    states[val_indices, :], g_text[val_indices, ], e_text[val_indices, ], n_assumptions[val_indices, ], a_vectors[val_indices, :]
states_test, g_text_test, e_text_test, n_assumptions_test, y_test = \
    states[test_indices, :], g_text[test_indices, ], e_text[test_indices, ], n_assumptions[test_indices, ], a_vectors[test_indices, :]
assert states_train.shape[0] == g_text_train.shape[0] == e_text_train.shape[0] == n_assumptions_train.shape[0] == y_train.shape[0]
assert states_val.shape[0] == g_text_val.shape[0] == e_text_val.shape[0] == n_assumptions_val.shape[0] == y_val.shape[0]
assert states_test.shape[0] == g_text_test.shape[0] == e_text_test.shape[0] == n_assumptions_test.shape[0] == y_test.shape[0]

# Scale the states data
scaler = CustomMinMaxScaler()
states_train_scaled = scaler.scale(states_train)  # The scaler will be only be fit on this call
states_val_scaled = scaler.scale(states_val)
states_test_scaled = scaler.scale(states_test)
scaler.save(f'../networks/scalers/{NETWORK_NAME}_state_scaler.pickle')

# early_stop = EarlyStopping(monitor='val_mean_squared_error', verbose=1, patience=EARLY_STOP)
# model_checkpoint = ModelCheckpoint(f'../networks/models/{NETWORK_NAME}.keras', monitor='val_mean_squared_error', save_best_only=True, verbose=1)
# history = model.fit(
#     [g_text_train.astype(object), e_text_train.astype(object), n_assumptions_train.reshape(-1, 1), states_train_scaled.reshape(states_train.shape[0], 1, -1)], y_train,
#     batch_size=BATCH_SIZE,
#     epochs=N_EPOCHS,
#     validation_data=([g_text_val.astype(object), e_text_val.astype(object), n_assumptions_val, states_val_scaled], y_val),
#     callbacks=[early_stop, model_checkpoint]
# )

vocab = list(set(words.words()))
foo = AATention(max_text_length=100, text_embedding_dim=64, vocab=vocab, mask_value=PAD_VAL, state_output_dim=128,
                use_pos_encodings=False, key_dim=128, final_dense_1_dim=128, final_dense_2_dim=64, dropout_rate=0.1)
foo1 = foo(g_text_train, e_text_train, states_train_scaled, n_assumptions_train)
print(foo1.shape)
print(foo.summary())
