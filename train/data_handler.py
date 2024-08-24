import numpy as np
from typing import Tuple
from utils.utils import PAD_VAL


def _get_n_generators(domain: str) -> int:
    if domain == 'jhg':
        return 16

    return 1


def _get_n_assumptions(domain: str, generator_name: str) -> float:
    if domain == 'jhg':
        return 0.59

    return 1.0


class DataHandler(object):
    @staticmethod
    def extract_domain_data(domain: str = 'jhg', augment_states: bool = False, augment_g_descriptions: bool = False,
                            augment_e_descriptions: bool = False) -> Tuple[np.array, np.array, np.array, np.array,
                                                                           np.array]:
        folder = f'../train/training_data/{domain}/'
        states, g_text, e_text, n_assumptions, a_vectors = [], [], [], [], []
        n_generators = _get_n_generators(domain)

        for generator_idx in range(n_generators):
            file_path = f'{folder}generator_{generator_idx}'
            game_states = np.genfromtxt(f'{file_path}_s.csv', delimiter=',', skip_header=0)
            g_descriptions = np.genfromtxt(f'{file_path}_gd.csv', delimiter='\n', skip_header=0, dtype=str)
            e_descriptions = np.genfromtxt(f'{file_path}_ed.csv', delimiter='\n', skip_header=0, dtype=str)
            n_generator_assumptions = _get_n_assumptions(domain, f'generator_{generator_idx}')
            alignment_vectors = np.genfromtxt(f'{file_path}_av.csv', delimiter=',', skip_header=0)

            states.append(game_states)
            g_text.append(g_descriptions)
            e_text.append(e_descriptions)
            n_assumptions.extend([n_generator_assumptions for _ in range(game_states.shape[0])])
            a_vectors.append(alignment_vectors)

        states = np.concatenate(states, axis=0)
        g_text = np.concatenate(g_text, axis=0)
        e_text = np.concatenate(e_text, axis=0)
        n_assumptions = np.array(n_assumptions)
        a_vectors = np.concatenate(a_vectors, axis=0)

        if augment_states:
            new_states, new_g_text, new_e_text, new_n_assumptions, new_a_vectors = [], [], [], [], []

            for i in range(states.shape[0]):
                row = states[i]
                no_padding_vals = row[row != PAD_VAL]
                no_padding_vals = np.random.permutation(no_padding_vals)
                new_row = np.concatenate([no_padding_vals, np.full(states.shape[1] - no_padding_vals.shape[0], PAD_VAL)])
                new_states.append(new_row)
                new_g_text.append(g_text[i])
                new_e_text.append(e_text[i])
                new_n_assumptions.append(n_assumptions[i])
                new_a_vectors.append(a_vectors[i])

            states = np.concatenate([states, new_states])
            g_text = np.concatenate([g_text, new_g_text])
            e_text = np.concatenate([e_text, new_e_text])
            n_assumptions = np.concatenate([n_assumptions, new_n_assumptions])
            a_vectors = np.concatenate([a_vectors, new_a_vectors])

        assert states.shape[0] == g_text.shape[0] == e_text.shape[0] == n_assumptions.shape[0] == a_vectors.shape[0]

        return states, g_text, e_text, n_assumptions, a_vectors
