from collections import Counter
import matplotlib.pyplot as plt
from networks.attention import AATention
import nltk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from train.custom_min_max_scaler import CustomMinMaxScaler
from train.data_handler import DataHandler
from utils.utils import PAD_VAL
nltk.download('words')
from nltk.corpus import words

# TODO:
#   - Friday:
#       - Try removing position encodings from text layer
#       - Try adding positional encodings to state layer
#       - Try adding dense layers after attention
#       - Try adding dropout to text layer
#           - If this boost performance, try adding dropout to a few other places

# Parameters and items needed for training and testing
TRAIN_PERCENTAGE = 0.7  # How much data to use for training
VALIDATION_PERCENTAGE = 0.3  # How much of the training data to use for validation
NETWORK_NAME = 'AATention'
N_EPOCHS = 100
EARLY_STOP = int(N_EPOCHS * 0.2)
BATCH_SIZE = 16
optimizer = Adam()
loss_fn = MeanSquaredError()
val_metric = MeanSquaredErrorMetric()
test_metric = MeanSquaredErrorMetric()
best_val_mse = np.inf
n_epochs_without_change = 0

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

# Create the model
vocab = words.words()
freq_dist = Counter(vocab)
vocab = [word for word, _ in freq_dist.most_common(5000)]
model = AATention(max_text_length=100, text_embedding_dim=64, vocab=vocab, mask_value=PAD_VAL, state_output_dim=128,
                  use_pos_encodings=False, key_dim=128, final_dense_1_dim=128, final_dense_2_dim=64, dropout_rate=0.1)

# Lists to monitor training and validation losses over time - used to generate a plot at the end of this file
training_losses, validation_losses = [], []


# Train
def _mask_output(y_true, y_pred):  # Function to mask padded output values (for loss and metrics)
    mask = tf.not_equal(y_true, PAD_VAL)
    tf_mask = tf.cast(mask, tf.float32)

    return y_true * tf_mask, y_pred * tf_mask


n_batches, n_val_batches = len(states_train_scaled) // BATCH_SIZE, len(states_val) // BATCH_SIZE

for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_g_text, curr_e_text, curr_state, curr_n_assumptions, curr_y = g_text_train[start_idx:end_idx, ], \
                                                                           e_text_train[start_idx:end_idx, ], \
                                                                           states_train_scaled[start_idx:end_idx, :], \
                                                                           n_assumptions_train[start_idx:end_idx, ], \
                                                                           y_train[start_idx:end_idx, :]

        with tf.GradientTape() as tape:
            predictions = model(curr_g_text, curr_e_text, curr_state, curr_n_assumptions, training=True)
            loss = loss_fn(*_mask_output(curr_y, predictions))

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Once all batches for the epoch are complete, calculate the validation loss
    for i in range(n_val_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        curr_g_text, curr_e_text, curr_state, curr_n_assumptions, curr_y = g_text_val[start_idx:end_idx, ], \
                                                                           e_text_val[start_idx:end_idx, ], \
                                                                           states_val_scaled[start_idx:end_idx, :], \
                                                                           n_assumptions_val[start_idx:end_idx, ], \
                                                                           y_val[start_idx:end_idx, :]
        val_predictions = model(curr_g_text, curr_e_text, curr_state, curr_n_assumptions)
        val_metric.update_state(*_mask_output(curr_y, val_predictions))

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
        model.save(f'../networks/models/{NETWORK_NAME}.keras')

    else:
        # Increment the number of epochs that have passed without any improvement/change
        n_epochs_without_change += 1

        # If sufficient epochs have passed without improvement, cancel the training process
        if n_epochs_without_change >= EARLY_STOP:
            print(f'EARLY STOPPING - {n_epochs_without_change} HAVE PASSED WITHOUT VALIDATION IMPROVEMENT')
            break

# Once the training process is complete, calculate performance on the test set
model = load_model(f'../networks/models/{NETWORK_NAME}.keras')
n_test_batches = len(states_test_scaled) // BATCH_SIZE

# for i in range(n_test_batches):
#     start_idx = i * BATCH_SIZE
#     end_idx = start_idx + BATCH_SIZE
#
#     curr_g_text, curr_e_text, curr_state, curr_n_assumptions, curr_y = g_text_test[start_idx:end_idx, ], \
#                                                                        e_text_test[start_idx:end_idx, ], \
#                                                                        states_test_scaled[start_idx:end_idx, :], \
#                                                                        n_assumptions_test[start_idx:end_idx, ], \
#                                                                        y_test[start_idx:end_idx, :]
#     test_predictions = model(curr_g_text, curr_e_text, curr_state, curr_n_assumptions)
#     test_metric.update_state(*_mask_output(curr_y, test_predictions))

test_predictions = model(g_text_test, e_text_test, states_test_scaled, n_assumptions_test)
test_metric.update_state(*_mask_output(y_test, test_predictions))

test_mse = test_metric.result()
print(f'Test MSE = {test_mse}')

# Save the test predictions and true values (for future analysis)
pred_vals_path = f'../analysis/test_data_results/{NETWORK_NAME}_test_pred_vals.csv'
true_vals_path = f'../analysis/test_data_results/{NETWORK_NAME}_test_true_vals.csv'
with open(pred_vals_path, 'wb') as file1, open(true_vals_path, 'wb') as file2:
    pickle.dump(test_predictions.numpy(), file1)
    pickle.dump(y_test, file2)

# Generate a plot of the training and validation loss over time
assert len(training_losses) == len(validation_losses)
x_vals = list(range(len(training_losses)))
plt.plot(x_vals, training_losses, label='Losses')
plt.plot(x_vals, validation_losses, color='red', label='Validation Losses')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(f'../train/figures/{NETWORK_NAME}.png', bbox_inches='tight')
plt.clf()

print(model.summary())

# Validation MSE = 0.037975139915943146
# Test MSE = 0.09751230478286743
