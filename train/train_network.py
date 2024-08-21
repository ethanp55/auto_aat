import numpy as np
from train.custom_min_max_scaler import CustomMinMaxScaler
from train.data_handler import DataHandler

# TODO:
#   - Wednesday: start setting up initial network architecture
#   - Thursday: finish initial network architecture
#   - Thursday: debug
#   - Friday: start analyzing results

# Parameters
TRAIN_PERCENTAGE = 0.7  # How much data to use for training
VALIDATION_PERCENTAGE = 0.3  # How much of the training data to use for validation
NETWORK_NAME = 'attention'

# Grab the data
states, g_text, e_text, a_vectors = DataHandler.extract_domain_data(domain='jhg', augment_states=False, n_generators=1)

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
states_train, g_text_train, e_text_train, y_train = \
    states[train_indices, :], g_text[train_indices, ], e_text[train_indices, ], a_vectors[train_indices, :]
states_val, g_text_val, e_text_val, y_val = \
    states[val_indices, :], g_text[val_indices, ], e_text[val_indices, ], a_vectors[val_indices, :]
states_test, g_text_test, e_text_test, y_test = \
    states[test_indices, :], g_text[test_indices, ], e_text[test_indices, ], a_vectors[test_indices, :]
assert states_train.shape[0] == g_text_train.shape[0] == e_text_train.shape[0] == y_train.shape[0]
assert states_val.shape[0] == g_text_val.shape[0] == e_text_val.shape[0] == y_val.shape[0]
assert states_test.shape[0] == g_text_test.shape[0] == e_text_test.shape[0] == y_test.shape[0]

# Scale the states data
scaler = CustomMinMaxScaler()
states_train_scaled = scaler.scale(states_train)  # The scaler will be only be fit on this call
states_val_scaled = scaler.scale(states_val)
states_test_scaled = scaler.scale(states_test)
scaler.save(f'../train/scalers/{NETWORK_NAME}_state_scaler.pickle')
