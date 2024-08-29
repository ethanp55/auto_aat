from games.game_state import GameState
from simple_rl.agents.AgentClass import Agent
from utils.utils import P1, P2
import numpy as np
import random
from pulp import *
import json


class MinimaxAgent(Agent):
    def __init__(self, name: str, n_iterations: int, initial_state: GameState, file_name, explore_prob=0.2, gamma=0.95,
                 decay=0.9999954, read_from_file=False, player=None) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.n_iterations = n_iterations
        self.explore_prob = explore_prob
        self.gamma = gamma
        self.decay = decay
        self.p1_policy = {}
        self.p1_attack_policy = {}
        self.v_p1 = {}
        self.p2_policy = {}
        self.p2_attack_policy = {}
        self.v_p2 = {}
        self.player = player

        if read_from_file:
            self._read_from_file(file_name)

        else:
            self._train(initial_state)

            with open('../agents/files/minimax/' + str(file_name) + '_p1_policy.json', 'w') as fp:
                json.dump(self.p1_policy, fp)

            with open('../agents/files/minimax/' + str(file_name) + '_p1_attack_policy.json', 'w') as fp:
                json.dump(self.p1_attack_policy, fp)

            with open('../agents/files/minimax/' + str(file_name) + '_v_p1.json', 'w') as fp:
                json.dump(self.v_p1, fp)

            with open('../agents/files/minimax/' + str(file_name) + '_p2_policy.json', 'w') as fp:
                json.dump(self.p2_policy, fp)

            with open('../agents/files/minimax/' + str(file_name) + '_p2_attack_policy.json', 'w') as fp:
                json.dump(self.p2_attack_policy, fp)

            with open('../agents/files/minimax/' + str(file_name) + '_v_p2.json', 'w') as fp:
                json.dump(self.v_p2, fp)

    def act(self, state, reward, round_num):
        p1_available_actions, p2_available_actions = state.get_available_actions()

        state_policy_p1 = self.p1_policy.get(str(state))

        if state_policy_p1 is None:
            state_policy_p1 = [1 / len(p1_available_actions)] * len(p1_available_actions)

        state_policy_p2 = self.p2_policy.get(str(state))

        if state_policy_p2 is None:
            state_policy_p2 = [1 / len(p2_available_actions)] * len(p2_available_actions)

        action_p1 = random.choices(p1_available_actions, weights=state_policy_p1)[0]
        action_p2 = random.choices(p2_available_actions, weights=state_policy_p2)[0]

        return action_p1, action_p2

    def _read_from_file(self, file_name):
        with open('../agents/files/minimax/' + str(file_name) + '_p1_policy.json', 'r') as fp:
            self.p1_policy = json.load(fp)

        with open('../agents/files/minimax/' + str(file_name) + '_p1_attack_policy.json', 'r') as fp:
            self.p1_attack_policy = json.load(fp)

        with open('../agents/files/minimax/' + str(file_name) + '_v_p1.json', 'r') as fp:
            self.v_p1 = json.load(fp)

        with open('../agents/files/minimax/' + str(file_name) + '_p2_policy.json', 'r') as fp:
            self.p2_policy = json.load(fp)

        with open('../agents/files/minimax/' + str(file_name) + '_p2_attack_policy.json', 'r') as fp:
            self.p2_attack_policy = json.load(fp)

        with open('../agents/files/minimax/' + str(file_name) + '_v_p2.json', 'r') as fp:
            self.v_p2 = json.load(fp)

    def _train(self, initial_state: GameState):
        q_vals_p1 = {}
        alpha_p1 = 1.0
        q_vals_p2 = {}
        alpha_p2 = 1.0

        def rec_search(curr_state: GameState):
            nonlocal alpha_p1
            nonlocal alpha_p2

            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            explore_p1 = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])
            explore_p2 = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

            p1_available_actions, p2_available_actions = curr_state.get_available_actions()

            if explore_p1:
                action_p1 = random.choice(p1_available_actions)

            else:
                state_policy = self.p1_policy.get(str(curr_state), None)

                if state_policy is None:
                    state_policy = [1 / len(p1_available_actions)] * len(p1_available_actions)
                    self.p1_policy[str(curr_state)] = state_policy

                action_p1 = random.choices(p1_available_actions, weights=state_policy)[0]

            if explore_p2:
                action_p2 = random.choice(p2_available_actions)

            else:
                state_policy = self.p2_policy.get(str(curr_state), None)

                if state_policy is None:
                    state_policy = [1 / len(p2_available_actions)] * len(p2_available_actions)
                    self.p2_policy[str(curr_state)] = state_policy

                action_p2 = random.choices(p2_available_actions, weights=state_policy)[0]

            next_state = curr_state.next(action_p1, action_p2)

            reward_p1, reward_p2 = rec_search(next_state)

            # PLAYER 1
            q_vals_p1[(str(curr_state), action_p1, action_p2)] = (1 - alpha_p1) * q_vals_p1.get((str(curr_state), action_p1, action_p2), 1.0) + alpha_p1 * (reward_p1 + self.gamma * self.v_p1.get(str(next_state), 1.0))

            model = LpProblem("P1-Policy", LpMaximize)
            m = pulp.LpVariable("m")
            model += pulp.lpSum([m]), "Maximize_Minimum"
            action_probs = [str(i) for i in range(1, len(p1_available_actions) + 1)]
            lp_variables = LpVariable.matrix("X", action_probs, lowBound=0.0, upBound=1.0)
            allocation = np.array(lp_variables).reshape(-1, 1)
            possibility_matrix = np.array([[q_vals_p1.get((str(curr_state), our_action, opp_action), 1.0) for our_action in p1_available_actions] for opp_action in p2_available_actions])

            for i in range(len(possibility_matrix)):
                label = 'Min constraint ' + str(i)
                curr_sum = sum([possibility_matrix[i][j] * allocation[j][0] for j in range(len(allocation))])
                condition = lpSum([m]) <= curr_sum
                model += condition, label

            model += lpSum(allocation) == 1.0
            model.solve(PULP_CBC_CMD(msg=False))

            new_policy = []

            for v in model.variables():
                try:
                    if 'X' in v.name:
                        new_policy.append(v.value())

                except:
                    print("Could not find LP variable value")

            self.p1_policy[str(curr_state)] = new_policy

            min_val = np.inf
            attack_action = None

            for opp_action in p2_available_actions:
                curr_opp_sum = 0

                for action in p1_available_actions:
                    state_policy = self.p1_policy[str(curr_state)]

                    curr_opp_sum += state_policy[p1_available_actions.index(action)] * q_vals_p1.get((str(curr_state), action, opp_action), 1.0)

                if curr_opp_sum < min_val:
                    min_val = curr_opp_sum
                    attack_action = opp_action

            self.v_p1[str(curr_state)] = min_val
            self.p2_attack_policy[str(curr_state)] = attack_action

            # PLAYER 2
            q_vals_p2[(str(curr_state), action_p2, action_p1)] = (1 - alpha_p2) * q_vals_p2.get((str(curr_state), action_p2, action_p1), 1.0) + alpha_p2 * (reward_p2 + self.gamma * self.v_p2.get(str(next_state), 1.0))

            model = LpProblem("P2-Policy", LpMaximize)
            m = pulp.LpVariable("m")
            model += pulp.lpSum([m]), "Maximize_Minimum"
            action_probs = [str(i) for i in range(1, len(p2_available_actions) + 1)]
            lp_variables = LpVariable.matrix("X", action_probs, lowBound=0.0, upBound=1.0)
            allocation = np.array(lp_variables).reshape(-1, 1)
            possibility_matrix = np.array([[q_vals_p2.get((str(curr_state), our_action, opp_action), 1.0) for our_action in p2_available_actions] for opp_action in p1_available_actions])

            for i in range(len(possibility_matrix)):
                label = 'Min constraint ' + str(i)
                curr_sum = sum([possibility_matrix[i][j] * allocation[j][0] for j in range(len(allocation))])
                condition = lpSum([m]) <= curr_sum
                model += condition, label

            model += lpSum(allocation) == 1.0
            model.solve(PULP_CBC_CMD(msg=False))

            new_policy = []

            for v in model.variables():
                try:
                    if 'X' in v.name:
                        new_policy.append(v.value())

                except:
                    print("Could not find LP variable value")

            self.p2_policy[str(curr_state)] = new_policy

            min_val = np.inf
            attack_action = None

            for opp_action in p1_available_actions:
                curr_opp_sum = 0

                for action in p2_available_actions:
                    state_policy = self.p2_policy.get(str(curr_state), None)

                    curr_opp_sum += state_policy[p2_available_actions.index(action)] * q_vals_p2.get((str(curr_state), action, opp_action), 1.0)

                if curr_opp_sum < min_val:
                    min_val = curr_opp_sum
                    attack_action = opp_action

            self.v_p2[str(curr_state)] = min_val
            self.p1_attack_policy[str(curr_state)] = attack_action

            alpha_p1 *= self.decay
            alpha_p2 *= self.decay

            return reward_p1, reward_p2

        for iter_num in range(self.n_iterations):
            print(f'Minimax q iteration: {iter_num + 1}')
            rec_search(initial_state)