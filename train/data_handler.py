import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from utils.utils import PAD_VAL, CHICKEN_E_DESCRIPTION


def _get_n_generators(domain: str) -> int:
    if domain == 'jhg':
        return 16

    elif domain == 'chicken':
        return 7

    return 1


def _get_n_assumptions(domain: str, generator_name: str) -> float:
    # if domain == 'jhg':
    #     return 0.59
    #
    # return 1.0
    return 0.59


def _int_to_str(generator_idx: int) -> str:
    idx_adj = generator_idx + 1

    if idx_adj == 1:
        num_str = 'one'

    elif idx_adj == 2:
        num_str = 'two'

    elif idx_adj == 3:
        num_str = 'three'

    elif idx_adj == 4:
        num_str = 'four'

    elif idx_adj == 5:
        num_str = 'five'

    elif idx_adj == 6:
        num_str = 'six'

    elif idx_adj == 7:
        num_str = 'seven'

    elif idx_adj == 8:
        num_str = 'eight'

    elif idx_adj == 9:
        num_str = 'nine'

    elif idx_adj == 10:
        num_str = 'ten'

    elif idx_adj == 11:
        num_str = 'eleven'

    elif idx_adj == 12:
        num_str = 'twelve'

    elif idx_adj == 13:
        num_str = 'thirteen'

    elif idx_adj == 14:
        num_str = 'fourteen'

    elif idx_adj == 15:
        num_str = 'fifteen'

    else:
        num_str = 'sixteen'

    return f'generator {num_str}'


def _other_e_descriptions(domain: str) -> str:
    # Repeated, multi-agent, collective-action, general-sum game where players attack or steal in order to build or
    # destroy relationships
    if domain == 'jhg':
        other_descriptions = ['General-sum game', 'General-sum social game', 'General-sum social dilemma',
                              'Collective action game with many players', 'Social game',
                              'Domain where agents build or destroy relationships', 'Human social interaction model']

    elif domain == 'chicken':
        other_descriptions = ['A daring, social game', 'A game where cooperation is possible but players must alternate between swerving and going straight']

    else:
        other_descriptions = []

    return np.random.choice(other_descriptions)


class DataHandler(object):
    @staticmethod
    def extract_domain_data(domain: str = 'jhg', augment_states: bool = False, augment_g_descriptions: bool = False,
                            augment_e_descriptions: bool = False) -> Tuple[np.array, np.array, np.array, np.array,
                                                                           np.array]:
        states, g_text, e_text, n_assumptions, a_vectors = [], [], [], [], []
        n_generators = _get_n_generators(domain)

        if domain == 'jhg':
            folder = f'../train/training_data/{domain}/'

            for generator_idx in range(n_generators):
                file_path = f'{folder}generator_{generator_idx}'
                game_states = np.genfromtxt(f'{file_path}_s.csv', delimiter=',', skip_header=0)
                # g_descriptions = np.genfromtxt(f'{file_path}_gd.csv', delimiter='\n', skip_header=0, dtype=str)
                g_descriptions = np.array([_int_to_str(generator_idx)] * len(game_states))
                e_descriptions = np.genfromtxt(f'{file_path}_ed.csv', delimiter='\n', skip_header=0, dtype=str)
                n_generator_assumptions = _get_n_assumptions(domain, f'generator_{generator_idx}')
                alignment_vectors = np.genfromtxt(f'{file_path}_av.csv', delimiter=',', skip_header=0)

                states.append(game_states)
                g_text.append(g_descriptions)
                e_text.append(e_descriptions)
                n_assumptions.extend([n_generator_assumptions for _ in range(game_states.shape[0])])
                a_vectors.append(alignment_vectors)

        elif domain == 'chicken':
            folder = f'../repeated_games/aat/training_data/{domain}_game/'
            generators = ['AlgaaterCoop', 'AlgaaterCoopPunish', 'AlgaaterBully', 'AlgaaterBullyPunish',
                          'AlgaaterBullied', 'AlgaaterMinimax', 'AlgaaterCfr']

            for generator_idx, generator_name in enumerate(generators):
                file_path = f'{folder}{generator_name}'
                game_states = np.array(pickle.load(open(f'{file_path}_states.pickle', 'rb')))
                indices = np.arange(game_states.shape[0])
                np.random.shuffle(indices)
                indices = indices[:int(game_states.shape[0] * 0.25)]
                game_states = game_states[indices, :]
                g_descriptions = np.array([_int_to_str(generator_idx)] * len(game_states))
                e_descriptions = np.array([CHICKEN_E_DESCRIPTION] * len(game_states))
                n_generator_assumptions = _get_n_assumptions(domain, f'generator_{generator_idx}')
                alignment_vectors = np.array(pickle.load(open(f'{file_path}_training_data.pickle', 'rb')))[indices, 0:-2]
                alignment_vectors = MinMaxScaler().fit_transform(alignment_vectors)
                assert alignment_vectors.min() == 0 and alignment_vectors.max() == 1
                n_padding = 100 - alignment_vectors.shape[1]
                alignment_vectors = np.concatenate([alignment_vectors, np.full((alignment_vectors.shape[0], n_padding), PAD_VAL)], axis=1)

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

        # Shuffle state feature orders
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

        # Add new text descriptions for the environment/domain
        if augment_e_descriptions:
            new_states, new_g_text, new_e_text, new_n_assumptions, new_a_vectors = [], [], [], [], []

            for i in range(states.shape[0]):
                new_states.append(states[i])
                new_g_text.append(g_text[i])
                new_env_text = _other_e_descriptions(domain)
                new_e_text.append(new_env_text)
                new_n_assumptions.append(n_assumptions[i])
                new_a_vectors.append(a_vectors[i])

            states = np.concatenate([states, new_states])
            g_text = np.concatenate([g_text, new_g_text])
            e_text = np.concatenate([e_text, new_e_text])
            n_assumptions = np.concatenate([n_assumptions, new_n_assumptions])
            a_vectors = np.concatenate([a_vectors, new_a_vectors])

        assert states.shape[0] == g_text.shape[0] == e_text.shape[0] == n_assumptions.shape[0] == a_vectors.shape[0]

        return states, g_text, e_text, n_assumptions, a_vectors
