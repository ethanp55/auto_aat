import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE
from typing import List


FILE_ADJ = 'forex'


def _plot_embeddings(labels: List[str], embeddings: np.array, agent_name: str, three_dimensions: bool = False) -> None:
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('tab20')(Normalize()([idx for idx in range(len(unique_labels))]))
    fig = plt.figure(figsize=(15, 15))
    plt.grid()

    if three_dimensions:
        ax = fig.add_subplot(111, projection='3d')

        for j, label in enumerate(unique_labels):
            label_points = embeddings[labels == label]
            ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], s=10, alpha=1.0, color=colors[j],
                       label=label)

    else:
        for j, label in enumerate(unique_labels):
            label_points = embeddings[labels == label]
            plt.scatter(label_points[:, 0], label_points[:, 1], s=10, alpha=1.0, color=colors[j], label=label)

    plt.legend(loc='best', fontsize='18')

    image_name_adj = '_3d' if three_dimensions else ''
    plt.savefig(f'../analysis/vector_plots/{agent_name}{image_name_adj}_{FILE_ADJ}.png', bbox_inches='tight')
    plt.clf()


folder = f'../analysis/{FILE_ADJ}_vectors/'
gen_vectors = {}
agents_to_include = ['AlegAATr', 'AlegAAATr', 'SMAlegAATr', 'AlegAAATTr']

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name not in agents_to_include:
        continue
    file_path = f'{folder}{file}'
    data = pd.read_csv(file_path, header=None)
    generators, vectors = np.array(data.iloc[:, 0]), np.array(data.iloc[:, 1:], dtype=float)
    assert generators.shape[0] == vectors.shape[0]

    for i in range(generators.shape[0]):
        generator_name = generators[i]
        agent_data = gen_vectors.get(agent_name, {})
        agent_data[generator_name] = agent_data.get(generator_name, []) + [vectors[i, :]]
        gen_vectors[agent_name] = agent_data

for agent_name, agent_data in gen_vectors.items():
    print(agent_name)
    all_labels, all_vectors = [], None

    for key, vector_list in agent_data.items():
        all_labels.extend([key] * len(vector_list))
        all_vectors = np.array(vector_list) if all_vectors is None else np.concatenate(
            [all_vectors, np.array(vector_list)])

    for three_dimensions in [True, False]:
        n_components = 3 if three_dimensions else 2
        all_embeddings = TSNE(n_components=n_components).fit_transform(all_vectors)
        _plot_embeddings(all_labels, all_embeddings, agent_name, three_dimensions=three_dimensions)
