from repeated_games.game_state import GameState
from simple_rl.agents.AgentClass import Agent
from utils.utils import P1, P2
from typing import List
import numpy as np
import jsonpickle


class CFRGameNode(object):
    def __init__(self, game_state: GameState, available_actions: List[str]):
        self.game_state = game_state
        self.available_actions = available_actions
        self.regret_sum = {}
        self.strategy = {}
        self.strategy_sum = {}

    def get_strategy(self, realization_weight: float):
        normalizing_sum = 0

        for action in self.available_actions:
            self.strategy[action] = max(self.regret_sum.get(action, 0), 0)
            normalizing_sum += self.strategy[action]

        for action in self.available_actions:
            if normalizing_sum > 0:
                self.strategy[action] /= normalizing_sum

            else:
                self.strategy[action] = 1.0 / len(self.available_actions)

            self.strategy_sum[action] = self.strategy_sum.get(action, 0) + realization_weight * self.strategy[action]

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = {}
        normalizing_sum = 0

        for action in self.available_actions:
            normalizing_sum += self.strategy_sum[action]

        for action in self.available_actions:
            if normalizing_sum > 0:
                avg_strategy[action] = self.strategy_sum[action] / normalizing_sum

            else:
                avg_strategy[action] = 1.0 / len(self.available_actions)

        return avg_strategy

    def __str__(self):
        return str(self.game_state) + '_' + str(self.get_average_strategy())


class CFRAgent(Agent):
    def __init__(self, name: str, initial_game_state: GameState, n_iterations: int, file_name, read_from_file=False,
                 player=None) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.player = player

        if not read_from_file:
            if initial_game_state.is_simultaneous():
                self.policy = self._calculate_policy_simultaneous(initial_game_state, n_iterations)

            else:
                self.policy = self._calculate_policy(initial_game_state, n_iterations)

            with open('../agents/files/cfr/' + str(file_name) + '_policy.json', 'w') as fp:
                fp.write(jsonpickle.encode(self.policy))

        else:
            with open('../agents/files/cfr/' + str(file_name) + '_policy.json', 'r') as fp:
                self.policy = jsonpickle.decode(fp.read())

    def _calculate_policy_simultaneous(self, initial_game_state: GameState, n_iterations: int):
        node_map = {P1: {}, P2: {}}

        def _cfr(curr_state, p0: float, p1: float):
            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            p1_available_actions, p2_available_actions = curr_state.get_available_actions()

            if curr_state.is_stochastic:
                sampled_action1_idx = np.random.choice([i for i in range(len(p1_available_actions))])
                sampled_action1 = p1_available_actions[sampled_action1_idx]
                sampled_action2_idx = np.random.choice([i for i in range(len(p2_available_actions))])
                sampled_action2 = p2_available_actions[sampled_action2_idx]
                next_state = curr_state.next(sampled_action1, sampled_action2)

                return _cfr(next_state, p0, p1)

            node1 = node_map[P1].get(str(curr_state), None)
            node2 = node_map[P2].get(str(curr_state), None)

            if node1 is None:
                node1 = CFRGameNode(curr_state, p1_available_actions)
                node_map[P1][str(curr_state)] = node1

            if node2 is None:
                node2 = CFRGameNode(curr_state, p2_available_actions)
                node_map[P2][str(curr_state)] = node2

            nodes = {P1: node1, P2: node2}
            node_utils = {P1: 0, P2: 0}

            for player_key, node in nodes.items():
                strategy = node.get_strategy(realization_weight=p0 if player_key == P1 else p1)
                util = {}

                player_available_actions = p1_available_actions if player_key == P1 else p2_available_actions
                opp_available_actions = p2_available_actions if player_key == P1 else p1_available_actions

                for action in player_available_actions:
                    opp_action_idx = np.random.choice([opp_available_actions.index(val) for val in opp_available_actions])
                    opp_action = opp_available_actions[opp_action_idx]
                    next_state = curr_state.next(action, opp_action) if player_key == P1 else curr_state.next(opp_action, action)
                    util1, util2 = _cfr(next_state, p0 * strategy[action], p1) if player_key == P1 else _cfr(next_state, p0, p1 * strategy[action])
                    util[action] = util1 if player_key == P1 else util2
                    node_utils[player_key] += strategy[action] * util[action]

                for action in player_available_actions:
                    regret = util[action] - node_utils[player_key]
                    node.regret_sum[action] = node.regret_sum.get(action, 0) + (p1 if player_key == P1 else p0) * regret

            return node_utils[P1], node_utils[P2]

        training_util1, training_util2 = 0, 0
        for n in range(1, n_iterations + 1):
            print('Simultaneous CFR iteration #: ' + str(n))
            p1_util, p2_util = _cfr(initial_game_state, 1, 1)
            training_util1 += p1_util
            training_util2 += p2_util

        print('Simultaneous CFR average P1 util: ' + str(training_util1 / n_iterations))
        print('Simultaneous CFR average P2 util: ' + str(training_util2 / n_iterations))

        return node_map

    def _calculate_policy(self, initial_game_state: GameState, n_iterations: int):
        node_map = {P1: {}, P2: {}}

        def _cfr(curr_state, p0: float, p1: float):
            curr_turn = curr_state.turn
            p1_available_actions, p2_available_actions = curr_state.get_available_actions()

            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            elif curr_state.is_stochastic:
                sampled_action1_idx = np.random.choice([i for i in range(len(p1_available_actions))])
                sampled_action1 = p1_available_actions[sampled_action1_idx]
                sampled_action2_idx = np.random.choice([i for i in range(len(p2_available_actions))])
                sampled_action2 = p2_available_actions[sampled_action2_idx]
                next_state = curr_state.next(sampled_action1, sampled_action2)

                return _cfr(next_state, p0, p1)

            node = node_map[curr_turn].get(str(curr_state), None)

            if node is None:
                node = CFRGameNode(curr_state, p1_available_actions if curr_turn == P1 else p2_available_actions)
                node_map[curr_turn][str(curr_state)] = node

            node_util = 0

            strategy = node.get_strategy(realization_weight=p0 if curr_turn == P1 else p1)
            util = {}
            util1, util2 = 0, 0

            player_available_actions = p1_available_actions if curr_turn == P1 else p2_available_actions
            opp_available_actions = p2_available_actions if curr_turn == P1 else p1_available_actions

            for action in player_available_actions:
                opp_action_idx = np.random.choice([opp_available_actions.index(val) for val in opp_available_actions if val != action])
                opp_action = opp_available_actions[opp_action_idx]
                next_state = curr_state.next(action, opp_action) if curr_turn == P1 else curr_state.next(opp_action, action)
                new_util1, new_util2 = _cfr(next_state, p0 * strategy[action], p1) if curr_turn == P1 else _cfr(next_state, p0, p1 * strategy[action])
                util[action] = new_util1 if curr_turn == P1 else new_util2
                node_util += strategy[action] * util[action]
                util1 += new_util1
                util2 += new_util2

            for action in player_available_actions:
                regret = util[action] - node_util
                node.regret_sum[action] = node.regret_sum.get(action, 0) + (p1 if curr_turn == P1 else p0) * regret

            return (node_util, util2 / len(player_available_actions)) if curr_turn == P1 else (util1 / len(player_available_actions), node_util)

        training_util1, training_util2 = 0, 0
        for n in range(1, n_iterations + 1):
            print('CFR iteration #: ' + str(n))
            t_util1, t_util2 = _cfr(initial_game_state, 1, 1)
            training_util1 += t_util1
            training_util2 += t_util2

        print('CFR average util1: ' + str(training_util1 / n_iterations))
        print('CFR average util2: ' + str(training_util2 / n_iterations))

        return node_map

    def act(self, state: GameState, reward: float, round_num):
        p1_available_actions, p2_available_actions = state.get_available_actions()

        p1_policy = self.policy.get(str(P1), None)
        p1_policy_actions = p1_policy.get(str(state), None) if p1_policy is not None else None
        p1_indices = [i for i in range(len(p1_available_actions))]

        if p1_policy is None or p1_policy_actions is None:
            p1_action_idx = np.random.choice(p1_indices)

        else:
            p1_action_idx = np.random.choice(p1_indices, p=list(p1_policy_actions['strategy'].values()))

        p1_action = p1_available_actions[p1_action_idx]

        p2_policy = self.policy.get(str(P2), None)
        p2_policy_actions = p2_policy.get(str(state), None) if p2_policy is not None else None
        p2_indices = [i for i in range(len(p2_available_actions))]

        if p2_policy is None or p2_policy_actions is None:
            p2_action_idx = np.random.choice(p2_indices)

        else:
            p2_action_idx = np.random.choice(p2_indices, p=list(p2_policy_actions['strategy'].values()))

        p2_action = p2_available_actions[p2_action_idx]

        return p1_action, p2_action