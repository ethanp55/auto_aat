from stag_hare.agents.agent import Agent
from stag_hare.agents.generator_pool import GeneratorPool
from stag_hare.environment.state import State
import keras
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from typing import Tuple
from utils.utils import PAD_VAL, STAG_HARE_E_DESCRIPTION, STAG_HARE_G_DESCRIPTIONS


@keras.saving.register_keras_serializable()
class AutoAATHead(Model):
    def __init__(self, aat_dim: int = 100, state_dim: int = 300) -> None:
        super(AutoAATHead, self).__init__()
        self.aat_dim = aat_dim
        self.state_dim = state_dim

        self.dense_aat_1 = Dense(32, activation='relu')

        self.output_layer = Dense(1, activation='linear')

    def get_config(self):
        return {'state_dim': self.state_dim, 'aat_dim': self.aat_dim}

    def call(self, aat_state, return_transformed_state: bool = False) -> tf.Tensor:
        x_aat = self.dense_aat_1(aat_state)

        if return_transformed_state:
            return x_aat

        return self.output_layer(x_aat)


@keras.saving.register_keras_serializable()
class AutoAATWithHead(Model):
    def __init__(self, auto_aat_model: Model, auto_aat_head: Model) -> None:
        super(AutoAATWithHead, self).__init__()
        self.auto_aat_model = auto_aat_model
        self.auto_aat_head = auto_aat_head

    def get_config(self):
        return {'auto_aat_model': keras.saving.serialize_keras_object(self.auto_aat_model),
                'auto_aat_head': keras.saving.serialize_keras_object(self.auto_aat_head)}

    @classmethod
    def from_config(cls, config):
        auto_aat_model_config = config.pop('auto_aat_model')
        auto_aat_model = keras.saving.deserialize_keras_object(auto_aat_model_config)
        config['auto_aat_model'] = auto_aat_model

        auto_aat_head_config = config.pop('auto_aat_head')
        auto_aat_head = keras.saving.deserialize_keras_object(auto_aat_head_config)
        config['auto_aat_head'] = auto_aat_head

        return cls(**config)

    def call(self, data_tup: Tuple[np.array, np.array, np.array], training: bool = False,
             return_transformed_state: bool = False) -> tf.Tensor:
        aat_vec = self.auto_aat_model(data_tup, training=training)

        if return_transformed_state:
            return aat_vec

        return self.auto_aat_head(aat_vec)


class SMAlegAATr(Agent):
    def __init__(self, name: str = 'SMAlegAATr', train: bool = False, enhanced: bool = False) -> None:
        Agent.__init__(self, name)
        self.generator_pool = GeneratorPool(name, check_assumptions=True, no_baseline_labels=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        file_adj = '_enh' if enhanced else ''
        self.model = load_model(f'../aat/auto_aat_tuned/tuned_model{file_adj}.keras')
        self.scaler = pickle.load(open(f'../aat/auto_aat_tuned/tuned_scaler{file_adj}.pickle', 'rb'))
        self.train = train
        self.tracked_vector = None
        self.prev_reward = None

    def act(self, state: State, reward: float, round_num: int) -> Tuple[int, int]:
        self.prev_reward = reward

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num, self.generator_to_use_idx)

        # State vector
        curr_state = list(state.vector_representation(self.name))
        curr_state += [PAD_VAL] * (300 - len(curr_state))
        curr_state = np.array(curr_state)
        curr_state = self.scaler.transform(curr_state.reshape(1, -1))

        # Make predictions for each generator
        best_pred = -np.inf
        best_indices = []

        for generator_idx in self.generator_indices:
            g_description = STAG_HARE_G_DESCRIPTIONS[generator_idx]
            pred = self.model((np.array(g_description).reshape(1, -1),
                               np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                               curr_state)).numpy()[0][0]

            if pred == best_pred:
                best_indices.append(generator_idx)

            elif pred > best_pred:
                best_pred = pred
                best_indices = [generator_idx]

        best_generator_idx = np.random.choice(best_indices)
        self.generator_to_use_idx = best_generator_idx
        best_desc = STAG_HARE_G_DESCRIPTIONS[self.generator_to_use_idx]
        self.tracked_vector = self.model((np.array(best_desc).reshape(1, -1),
                                          np.array(STAG_HARE_E_DESCRIPTION).reshape(1, -1),
                                          curr_state),
                                         return_transformed_state=True).numpy().reshape(-1, )

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        # If we're done and are supposed to train AAT, do so
        done = state.hare_captured() or state.stag_captured()
        if done and self.train:
            self.generator_pool.train_aat(enhanced=True)

        return token_allocations
