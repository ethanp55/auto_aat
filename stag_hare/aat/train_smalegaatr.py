from repeated_games.agents.smalegaatr import AutoAATHead, AutoAATWithHead
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from utils.utils import NETWORK_NAME, STAG_HARE_E_DESCRIPTION, STAG_HARE_G_DESCRIPTIONS


ENHANCED = True
adjustment = '_enh' if ENHANCED else ''
N_EPOCHS = 50
VALIDATION_PERCENTAGE = 0.3
EARLY_STOP = int(N_EPOCHS * 0.1)
BATCH_SIZE = 16
training_data_folder = '../aat/training_data/'
states, labels, g_text = None, None, []
DESCRIPTION = STAG_HARE_E_DESCRIPTION

for file in os.listdir(training_data_folder):
    if ('sin_c' not in file) or (ENHANCED and '_enh' not in file) or (not ENHANCED and '_enh' in file) \
            or 'vectors' in file:
        continue

    data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
    is_state = 'states' in file

    if is_state:
        gen_idx = int(file.split('_')[1])
        g_text.extend([STAG_HARE_G_DESCRIPTIONS[gen_idx]] * data.shape[0])

        if states is None:
            states = data

        else:
            states = np.concatenate([states, data])

    else:
        if labels is None:
            labels = data

        else:
            labels = np.concatenate([labels, data])

# Create training and validation sets
g_text = np.array(g_text)
assert states.shape[0] == labels.shape[0] == g_text.shape[0]
n_samples = states.shape[0]
n_val = int(n_samples * VALIDATION_PERCENTAGE)
indices = np.arange(n_samples)
np.random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]
assert len(val_indices) == n_val
assert len(train_indices) == n_samples - n_val
states_train, y_train = states[train_indices, :], labels[train_indices,]
g_text_train, e_text_train = g_text[train_indices, ], np.array([DESCRIPTION] * len(states_train))
states_val, y_val = states[val_indices, :], labels[val_indices,]
g_text_val, e_text_val = g_text[val_indices], np.array([DESCRIPTION] * len(states_val))
assert states_train.shape[0] == g_text_train.shape[0] == e_text_train.shape[0] == y_train.shape[0]
assert states_val.shape[0] == g_text_val.shape[0] == e_text_val.shape[0] == y_val.shape[0]
print(states_train.shape, g_text_train.shape, e_text_train.shape, y_train.shape)
print(states_val.shape, g_text_val.shape, e_text_val.shape, y_val.shape)

# State scaler
scaler = StandardScaler()
states_train = scaler.fit_transform(states_train)
states_val = scaler.transform(states_val)
with open(f'../aat/auto_aat_tuned/tuned_scaler{adjustment}.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Create model
pretrained_aat_gen_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
auto_aat_head = AutoAATHead()
model = AutoAATWithHead(pretrained_aat_gen_model, auto_aat_head)

# Items needed for training
optimizer = Adam()
loss_fn = MeanSquaredError()
val_metric = MeanSquaredErrorMetric()
test_metric = MeanSquaredErrorMetric()
best_val_mse = np.inf
n_epochs_without_change = 0

# Lists to monitor training and validation losses over time - used to generate a plot at the end of training
training_losses, validation_losses = [], []

# Batches
n_batches, n_val_batches = len(states_train) // BATCH_SIZE, len(states_val) // BATCH_SIZE

# Train
for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')

    # Iterate through training batches, tune model
    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_states, curr_g_text, curr_e_text, curr_y = states_train[start_idx:end_idx, :], \
                                                        g_text_train[start_idx:end_idx, ], \
                                                        e_text_train[start_idx:end_idx, ], \
                                                        y_train[start_idx:end_idx, ]

        with tf.GradientTape() as tape:
            predictions = model((curr_g_text, curr_e_text, curr_states), training=True)
            loss = loss_fn(curr_y, predictions)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Once all batches for the epoch are complete, calculate the validation loss
    for i in range(n_val_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_states, curr_g_text, curr_e_text, curr_y = \
            states_val[start_idx:end_idx, :], \
            g_text_val[start_idx:end_idx, ], \
            e_text_val[start_idx:end_idx, ], \
            y_val[start_idx:end_idx, ]

        val_predictions = model((curr_g_text, curr_e_text, curr_states))
        val_metric.update_state(curr_y, val_predictions)

    val_mse = val_metric.result()
    val_metric.reset_state()

    print(f'Train MSE = {loss}, Validation MSE = {val_mse}')

    # Add the training and validation losses to their corresponding lists
    training_losses, validation_losses = training_losses + [loss], validation_losses + [val_mse]

    # Make updates if the validation performance improved
    if val_mse < best_val_mse:
        print(f'Validation performance improved from {best_val_mse} to {val_mse}')

        # Reset the number of epochs that have passed without any improvement/change
        n_epochs_without_change = 0

        # Update the best validation performance metric
        best_val_mse = val_mse

        # Save the network
        model.save(f'../aat/auto_aat_tuned/tuned_model{adjustment}.keras')

    else:
        # Increment the number of epochs that have passed without any improvement/change
        n_epochs_without_change += 1

        # If sufficient epochs have passed without improvement, cancel the training process
        if n_epochs_without_change >= EARLY_STOP:
            print(f'EARLY STOPPING - {n_epochs_without_change} HAVE PASSED WITHOUT VALIDATION IMPROVEMENT')
            break

    # Shuffle the training data at the end of each epoch
    indices = np.arange(len(states_train))
    np.random.shuffle(indices)
    states_train, y_train = states_train[indices, :], y_train[indices,]
    g_text_train, e_text_train = g_text_train[indices,], e_text_train[indices,]
