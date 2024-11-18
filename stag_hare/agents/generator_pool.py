from stag_hare.agents.agent import Agent
from stag_hare.agents.generator import GreedyHareGen, GreedyPlannerHareGen, GreedyPlannerStagGen, TeamAwareGen
import csv
from stag_hare.environment.state import State
import fcntl
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from typing import List
from utils.utils import NETWORK_NAME, PAD_VAL, STAG_HARE_E_DESCRIPTION, STAG_HARE_G_DESCRIPTIONS


class GeneratorPool(Agent):
    def __init__(self, name: str, check_assumptions: bool = False, no_baseline_labels: bool = False) -> None:
        Agent.__init__(self, name)
        self.generators = []
        self.generators.append(GreedyHareGen(name, check_assumptions=check_assumptions))
        self.generators.append(GreedyPlannerHareGen(name, check_assumptions=check_assumptions))
        self.generators.append(GreedyPlannerStagGen(name, check_assumptions=check_assumptions))
        self.generators.append(TeamAwareGen(name, check_assumptions=check_assumptions))
        self.check_assumptions = check_assumptions
        self.generator_to_assumption_estimates = {}
        self.generators_to_states = {}
        self.no_baseline_labels = no_baseline_labels
        self.reward_history = []
        self.generator_just_used_idx = None
        self.prev_round_num = 0
        self.state = None

        # For training with AATention network
        self.assumption_pred_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
        self.state_scaler = pickle.load(open(f'../../networks/scalers/{NETWORK_NAME}_state_scaler.pickle', 'rb'))
        assert self.state_scaler._scaler is not None

        # For training with tuned AATention network (after training SMAlegAATr)
        self.pred_model = load_model('../aat/auto_aat_tuned/tuned_model_enh.keras')
        self.pred_scaler = pickle.load(open('../aat/auto_aat_tuned/tuned_scaler_enh.pickle', 'rb'))

    def act(self, state: State, reward: float, round_num: int, generator_just_used_idx: int):
        generator_to_action = {}

        for i, generator in enumerate(self.generators):
            generator_to_action[i] = generator.act(state, reward, round_num)

        if self.generator_just_used_idx is not None:
            generator_just_used = self.generators[self.generator_just_used_idx]

            # Grab the assumption estimates, if we're tracking them
            if self.check_assumptions:
                curr_state = list(self.state)
                curr_state += [PAD_VAL] * (300 - len(curr_state))

                # # AlegAATr (and SMAlegAATr, though it doesn't use these vectors)
                # assumps = self.assumptions(self.generator_just_used_idx)

                # # AlegAAATr
                # state_input_scaled = self.state_scaler.scale(np.array(curr_state).reshape(1, -1))
                # g_description = STAG_HARE_G_DESCRIPTIONS[self.generator_just_used_idx]
                # assumps = self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                #                                          np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                #                                          state_input_scaled)).numpy()[0]
                # assumps = list(assumps)

                # AlegAAATTr
                state_input_scaled = self.pred_scaler.transform(np.array(curr_state).reshape(1, -1))
                g_description = STAG_HARE_G_DESCRIPTIONS[self.generator_just_used_idx]
                assumps = self.pred_model.auto_aat_model((np.array(g_description).reshape(1, -1),
                                                             np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                                                             state_input_scaled)).numpy()[0]
                assumps = list(assumps)

                tup = (assumps, round_num, generator_just_used.baseline, None)

                self.generator_to_assumption_estimates[self.generator_just_used_idx] = \
                    self.generator_to_assumption_estimates.get(self.generator_just_used_idx, []) + [tup]

                if self.no_baseline_labels:
                    self.generators_to_states[self.generator_just_used_idx] = self.generators_to_states.get(
                        self.generator_just_used_idx, []) + [curr_state]

        self.reward_history.append(reward)
        self.prev_round_num = round_num

        # Check assumptions
        if self.check_assumptions:
            for i, generator in enumerate(self.generators):
                was_used = i == generator_just_used_idx
                generator.check_assumptions(state)
            self.generator_just_used_idx = generator_just_used_idx
            self.state = state.vector_representation(self.name)

        return generator_to_action

    def train_aat(self, discount_factor: float = 0.9, enhanced: bool = False) -> None:
        # Calculate discounted rewards
        discounted_rewards, running_sum = [0] * (len(self.reward_history) - 1), 0
        for i in reversed(range(len(self.reward_history))):
            if i == 0:
                break
            reward = self.reward_history[i] - self.reward_history[i - 1]
            running_sum = reward + discount_factor * running_sum
            discounted_rewards[i - 1] = running_sum

        # Store the training data
        for generator_idx in self.generator_to_assumption_estimates.keys():
            assumptions_history = self.generator_to_assumption_estimates[generator_idx]
            states_history = self.generators_to_states.get(generator_idx, [])

            for i in range(len(assumptions_history)):
                assumption_estimates, round_num, baseline, game_state = assumptions_history[i]
                state = states_history[i] if self.no_baseline_labels else None
                assert round_num > 0

                discounted_reward = discounted_rewards[round_num - 1]
                correction_term = discounted_reward / baseline
                alignment_vector = assumption_estimates

                if self.no_baseline_labels:
                    # Store the alignment vector
                    adjustment = '_enh' if enhanced else ''
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_vectors{adjustment}.csv'

                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(alignment_vector)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the state
                    assert state is not None
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_states{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(state)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the discounted reward
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_correction_terms{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow([discounted_reward])
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                else:
                    # Store the alignment vector
                    adjustment = '_enh' if enhanced else ''
                    file_path = f'../aat/training_data/generator_{generator_idx}_vectors{adjustment}.csv'

                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(alignment_vector)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the correction term
                    file_path = f'../aat/training_data/generator_{generator_idx}_correction_terms{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow([correction_term])
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

    def assumptions(self, generator_idx: int) -> List[float]:
        return self.generators[generator_idx].assumptions()
