import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.manifold import TSNE
from utils.utils import PAD_VAL


# Network name
NETWORK_NAME = 'AATention'

# Get data
pred_vals_path = f'../analysis/test_data_results/{NETWORK_NAME}_test_pred_vals.pickle'
true_vals_path = f'../analysis/test_data_results/{NETWORK_NAME}_test_true_vals.pickle'
y_hat = pickle.load(open(pred_vals_path, 'rb'))
y = pickle.load(open(true_vals_path, 'rb'))
y_hat = np.where(y_hat == PAD_VAL, 0.0, y_hat)
y = np.where(y == PAD_VAL, 0.0, y_hat)

# Make sure the number of predicted and true values are the same
n_points = y_hat.shape[0]
assert y.shape[0] == n_points

# Compress
combined = np.concatenate([y_hat, y], axis=0)
embeddings = TSNE(n_components=2).fit_transform(combined)
y_hat_embeddings = embeddings[:n_points, :]
y_embeddings = embeddings[n_points:, :]
assert y_hat_embeddings.shape == y_embeddings.shape

# Plot
plt.scatter(y_embeddings[:, 0], y_embeddings[:, 1], s=5, alpha=0.8, color='blue', label='True')
plt.scatter(y_hat_embeddings[:, 0], y_hat_embeddings[:, 1], s=5, alpha=0.8, color='red', label='Pred')
plt.legend(loc='best')
plt.savefig(f'../analysis/figures/{NETWORK_NAME}_preds_vs_true.png', bbox_inches='tight')
plt.clf()

plt.scatter(y_hat_embeddings[:, 0], y_hat_embeddings[:, 1], s=5, alpha=0.8, color='red', label='Pred')
plt.legend(loc='best')
plt.savefig(f'../analysis/figures/{NETWORK_NAME}_preds.png', bbox_inches='tight')
plt.clf()

plt.scatter(y_embeddings[:, 0], y_embeddings[:, 1], s=5, alpha=0.8, color='blue', label='True')
plt.legend(loc='best')
plt.savefig(f'../analysis/figures/{NETWORK_NAME}_true.png', bbox_inches='tight')
plt.clf()
