# Python imports
from collections.abc import Sequence

import numpy as np

''' GameState.py: Contains the State Class. '''


class GameState(Sequence):
    ''' Abstract State class '''

    def __init__(self, turn, data=[], is_stochastic=False, is_terminal=False):
        self.turn = turn
        self.data = data
        self._is_terminal = is_terminal
        self.is_stochastic = is_stochastic

    def features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have functiona
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return np.array(self.data).flatten()

    def get_data(self):
        return self.data

    def get_num_feats(self):
        return len(self.features())

    def is_terminal(self):
        return self._is_terminal

    def set_terminal(self, is_term=True):
        self._is_terminal = is_term

    def reward(self, player):
        return 0

    def next(self, action_0, action_1):
        return None

    def get_available_actions(self):
        return None, None

    def is_simultaneous(self):
        return self.turn is None

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.data))
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self.data == other.data
        return False

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)