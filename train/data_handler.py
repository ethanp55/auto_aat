import numpy as np
from typing import Tuple
from utils.utils import PAD_VAL


class DataHandler(object):
    @staticmethod
    def extract_domain_data(domain: str = 'jhg', n_generators: int = 16, augment_states: bool = False,
                            augment_g_descriptions: bool = False,
                            augment_e_descriptions: bool = False) -> Tuple[np.array, np.array, np.array, np.array]:
        folder = f'../train/training_data/{domain}/'
        states, g_text, e_text, a_vectors = [], [], [], []

        for generator_idx in range(n_generators):
            file_path = f'{folder}generator_{generator_idx}'
            game_states = np.genfromtxt(f'{file_path}_s.csv', delimiter=',', skip_header=0)
            g_descriptions = np.genfromtxt(f'{file_path}_gd.csv', delimiter='\n', skip_header=0, dtype=str)
            e_descriptions = np.genfromtxt(f'{file_path}_ed.csv', delimiter='\n', skip_header=0, dtype=str)
            alignment_vectors = np.genfromtxt(f'{file_path}_av.csv', delimiter=',', skip_header=0)

            states.append(game_states)
            g_text.append(g_descriptions)
            e_text.append(e_descriptions)
            a_vectors.append(alignment_vectors)

        states = np.concatenate(states, axis=0)
        a_vectors = np.concatenate(a_vectors, axis=0)
        g_text = np.concatenate(g_text, axis=0)
        e_text = np.concatenate(e_text, axis=0)

        if augment_states:
            new_states, new_a_vectors, new_g_text, new_e_text = [], [], [], []

            for i in range(states.shape[0]):
                row = states[i]
                no_padding_vals = row[row != PAD_VAL]
                no_padding_vals = np.random.permutation(no_padding_vals)
                new_row = np.concatenate([no_padding_vals, np.full(states.shape[1] - no_padding_vals.shape[0], PAD_VAL)])
                new_states.append(new_row)
                new_a_vectors.append(a_vectors[i])
                new_g_text.append(g_text[i])
                new_e_text.append(e_text[i])

            states = np.concatenate([states, new_states])
            a_vectors = np.concatenate([a_vectors, new_a_vectors])
            g_text = np.concatenate([g_text, new_g_text])
            e_text = np.concatenate([e_text, new_e_text])

        assert states.shape[0] == g_text.shape[0] == e_text.shape[0] == a_vectors.shape[0]

        return states, g_text, e_text, a_vectors
