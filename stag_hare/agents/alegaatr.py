from stag_hare.agents.agent import Agent
from stag_hare.agents.generator_pool import GeneratorPool
from collections import deque
from stag_hare.environment.state import State
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from typing import Tuple
from utils.utils import NETWORK_NAME, PAD_VAL, STAG_HARE_E_DESCRIPTION, STAG_HARE_G_DESCRIPTIONS


class AlegAATr(Agent):
    def __init__(self, name: str = 'AlegAATr', lmbda: float = 0.95, lookback: int = 5, train: bool = False,
                 enhanced: bool = False, auto_aat: bool = False, auto_aat_tuned: bool = False) -> None:
        Agent.__init__(self, name)
        self.auto_aat, self.auto_aat_tuned = auto_aat, auto_aat_tuned
        self.lmbda = lmbda
        self.generator_pool = GeneratorPool(name, check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        self.models, self.scalers = {}, {}
        self._read_in_generator_models(enhanced)
        self.empirical_increases, self.n_rounds_since_used = {}, {}
        self._initialize_empirical_data(lookback)
        self.prev_reward = None
        self.train = train
        self.tracked_vector = None

        if self.auto_aat:
            self.assumption_pred_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
            self.state_scaler = pickle.load(open(f'../../networks/scalers/{NETWORK_NAME}_state_scaler.pickle', 'rb'))
            assert self.state_scaler._scaler is not None

        elif self.auto_aat_tuned:
            adjustment = '_enh' if enhanced else ''
            self.assumption_pred_model = load_model(f'../aat/auto_aat_tuned/tuned_model{adjustment}.keras')
            self.state_scaler = pickle.load(open(f'../aat/auto_aat_tuned/tuned_scaler{adjustment}.pickle', 'rb'))

    def _read_in_generator_models(self, enhanced: bool) -> None:
        folder = '../aat/knn_models/'

        for file in os.listdir(folder):
            # Account for enhanced vs. not enhanced
            if (enhanced and '_enh' not in file) or (not enhanced and '_enh' in file):
                continue

            # Account for auto AAT, auto AAT tuned, and regular/plain AAT
            if self.auto_aat_tuned and 'auto_tuned' not in file:
                continue

            elif self.auto_aat and ('auto' not in file or 'tuned' in file):
                continue

            elif not self.auto_aat and not self.auto_aat_tuned and 'auto' in file:
                continue

            generator_idx = int(file.split('_')[1])
            full_file_path = f'{folder}{file}'

            if 'scaler' in file:
                self.scalers[generator_idx] = pickle.load(open(full_file_path, 'rb'))

            else:
                self.models[generator_idx] = pickle.load(open(full_file_path, 'rb'))

    def _initialize_empirical_data(self, lookback: int) -> None:
        for generator_idx in self.generator_indices:
            self.empirical_increases[generator_idx] = deque(maxlen=lookback)
            self.n_rounds_since_used[generator_idx] = 1

    def act(self, state: State, reward: float, round_num: int) -> Tuple[int, int]:
        # Update empirical results
        if self.prev_reward is not None:
            increase = reward - self.prev_reward
            self.empirical_increases[self.generator_to_use_idx].append(increase)
        self.prev_reward = reward

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num, self.generator_to_use_idx)

        # Make predictions for each generator
        best_pred, best_generator_idx, best_vector = -np.inf, None, None

        for generator_idx in self.generator_indices:
            if self.models.get(generator_idx, None) is None:
                continue
            n_rounds_since_last_use = self.n_rounds_since_used[generator_idx]
            use_emp_rewards = np.random.rand() < self.lmbda ** n_rounds_since_last_use and len(
                self.empirical_increases[generator_idx]) > 0

            # Use empirical results as the prediction
            if use_emp_rewards:
                increases = self.empirical_increases[generator_idx]
                avg = sum(increases) / len(increases)
                pred = avg

            # Otherwise, use AAT
            else:
                if self.auto_aat:
                    assert self.assumption_pred_model is not None
                    assert self.state_scaler is not None

                    state_input = list(state.vector_representation(self.name))
                    state_input += [PAD_VAL] * (300 - len(state_input))
                    state_input_scaled = self.state_scaler.scale(np.array(state_input).reshape(1, -1))
                    g_description = STAG_HARE_G_DESCRIPTIONS[generator_idx]
                    x = self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                                                    np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                                                    state_input_scaled)).numpy().reshape(1, -1)

                elif self.auto_aat_tuned:
                    state_input = list(state.vector_representation(self.name))
                    state_input += [PAD_VAL] * (300 - len(state_input))
                    state_input_scaled = self.state_scaler.transform(np.array(state_input).reshape(1, -1))
                    g_description = STAG_HARE_G_DESCRIPTIONS[generator_idx]
                    x = self.assumption_pred_model.auto_aat_model((np.array(g_description).reshape(1, -1),
                                                                   np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                                                                   state_input_scaled)).numpy()

                else:
                    generator_assumption_estimates = self.generator_pool.assumptions(generator_idx)
                    x = np.array(generator_assumption_estimates).reshape(1, -1)

                x_scaled = self.scalers[generator_idx].transform(x) if generator_idx in self.scalers else x
                correction_term_pred = self.models[generator_idx].predict(x_scaled)[0]
                pred = self.generator_pool.generators[generator_idx].baseline * correction_term_pred

            if pred > best_pred:
                best_pred, best_generator_idx = pred, generator_idx
                best_vector = x_scaled if not use_emp_rewards else None

        self.generator_to_use_idx = best_generator_idx

        if not self.auto_aat and not self.auto_aat_tuned:
            best_vector = best_vector.reshape(-1, 1)
            n_zeroes = 10 - best_vector.shape[0]
            best_vector = np.append(best_vector, np.zeros(n_zeroes)).reshape(1, -1)
        else:
            best_vector = best_vector.reshape(1, -1)
        self.tracked_vector = best_vector[0, :]

        # Update how many rounds it has been since each generator has been used
        for generator_idx in self.n_rounds_since_used.keys():
            if generator_idx == self.generator_to_use_idx:
                self.n_rounds_since_used[generator_idx] = 1

            else:
                self.n_rounds_since_used[generator_idx] += 1

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        # If we're done and are supposed to train AAT, do so
        done = state.hare_captured() or state.stag_captured()
        if done and self.train:
            self.generator_pool.train_aat(enhanced=True)

        return token_allocations
