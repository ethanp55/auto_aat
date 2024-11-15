from copy import deepcopy
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from repeated_games.agents.alegaatr import AlegAATr
from repeated_games.agents.folk_egal import FolkEgalPunishAgent
from repeated_games.chicken_game import ACTIONS as chicken_actions
from repeated_games.coordination_game import ACTIONS as coord_actions
from repeated_games.prisoners_dilemma import ACTIONS as pd_actions
import numpy as np
import pickle
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from typing import Tuple
from utils.utils import PRISONERS_E_DESCRIPTION, RG_G_DESCRIPTIONS, PAD_VAL


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

        return self.auto_aat_head(aat_vec, return_transformed_state=return_transformed_state)


class SMAlegAATr(Agent):
    def __init__(self, name, game: MarkovGameMDP, player: int) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.game = deepcopy(game)
        self.player = player
        self.generators = AlegAATr.create_aat_experts(self.game, self.player)
        self.generator_in_use_name = None
        self.tracked_vector = None
        self.model = load_model(f'../aat/auto_aat_tuned/{str(self.game)}_tuned_model.keras')
        self.scaler = pickle.load(open(f'../aat/auto_aat_tuned/{str(self.game)}_tuned_scaler.pickle', 'rb'))
        self.actions = pd_actions if str(game) == 'prisoners_dilemma_game' else \
            (chicken_actions if str(game) == 'chicken_game' else coord_actions)
        self.state = None
        self.round_switch_num = 0

    def update_state(self, state, prev_reward_1, prev_reward_2):
        state_input = [self.actions.index(state.actions[self.player]),
                       self.actions.index(state.actions[1 - self.player]),
                       prev_reward_1,
                       prev_reward_2]
        state_input += [PAD_VAL] * (300 - len(state_input))
        self.state = self.scaler.transform(np.array(state_input).reshape(1, -1))

    def act(self, state, reward, round_num):
        if round_num >= self.round_switch_num:
            old_expert = self.generator_in_use_name

            if self.state is None:
                state_input = [-1, -1, 0, 0]
                state_input += [PAD_VAL] * (300 - len(state_input))
                self.state = self.scaler.transform(np.array(state_input).reshape(1, -1))

            # Make predictions
            best_pred = -np.inf
            best_indices = []

            for generator_idx, generator in enumerate(list(self.generators.values())):
                g_description = RG_G_DESCRIPTIONS[generator.name]
                pred = self.model((np.array(g_description).reshape(1, -1),
                                   np.array(PRISONERS_E_DESCRIPTION).reshape(1, -1),
                                   self.state)).numpy()[0][0]
                if pred == best_pred:
                    best_indices.append(generator_idx)

                elif pred > best_pred:
                    best_pred = pred
                    best_indices = [generator_idx]

            best_generator_idx = np.random.choice(best_indices)
            self.generator_in_use_name = list(self.generators.keys())[best_generator_idx]
            best_desc = RG_G_DESCRIPTIONS[self.generator_in_use_name]
            self.tracked_vector = self.model((np.array(best_desc).reshape(1, -1),
                                              np.array(PRISONERS_E_DESCRIPTION).reshape(1, -1),
                                              self.state),
                                             return_transformed_state=True).numpy().reshape(-1, )

            self.round_switch_num = round_num + 2

            if old_expert != self.generator_in_use_name and isinstance(self.generators[self.generator_in_use_name], FolkEgalPunishAgent):
                self.generators[self.generator_in_use_name].start_round, self.generators[self.generator_in_use_name].should_attack = \
                    round_num + 1, False

        return self.generators[self.generator_in_use_name].act(state, reward, round_num)
