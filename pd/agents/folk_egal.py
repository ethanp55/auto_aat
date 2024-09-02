from pd.game_state import GameState
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from pd.agents.minimax_q import MinimaxAgent
from utils.utils import P1, P2
import numpy as np
import random
from pulp import *
import json
from copy import deepcopy


class FolkEgalPunishAgent(Agent):
    def __init__(self, name: str, generator: Agent, file_name: str, game: MarkovGameMDP):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.generator = generator
        self.player = generator.player
        self.game = deepcopy(game)
        self.should_attack = False
        self.start_round = 0

        minimax_agent = MinimaxAgent('FolkEgalPunishHelper', 0, generator.initial_state, file_name, read_from_file=True)

        self.p1_attack_policy = minimax_agent.p1_attack_policy
        self.p2_attack_policy = minimax_agent.p2_attack_policy

    def change_player(self, new_player):
        self.player = new_player
        self.generator.player = new_player

    def _find_if_should_attack(self, reward: float, round_num):
        self.should_attack = False
        rewards_1 = []
        rewards_2 = []

        for _ in range(5):
            self.game.reset()
            curr_state = deepcopy(self.game.get_init_state())
            reward_1, reward_2 = 0, 0
            action_map = dict()

            while not curr_state.is_terminal():
                action_p1, action_p2 = self.generator.act(curr_state, 0, round_num)

                action_map['1'], action_map['2'] = action_p1, action_p2

                updated_rewards_map, next_curr_state = self.game.execute_agent_action(action_map)

                for agent_name, new_reward in updated_rewards_map.items():
                    if agent_name == '1':
                        reward_1 += new_reward

                    else:
                        reward_2 += new_reward

                curr_state = next_curr_state

            rewards_1.append(reward_1)
            rewards_2.append(reward_2)

        rewards = rewards_1 if self.player == P1 else rewards_2
        min_reward, max_reward = min(rewards), max(rewards)

        self.should_attack = not min_reward <= reward <= max_reward

    def act(self, state, reward, round_num):
        p1_available_actions, p2_available_actions = state.get_available_actions()

        if round_num >= self.start_round + 1:
            if self.should_attack:
                self.should_attack = False

            else:
                self._find_if_should_attack(reward, round_num - 1)

            self.start_round = round_num

        if self.should_attack:
            action_p1, action_p2 = self.p1_attack_policy.get(str(state), None), self.p2_attack_policy.get(str(state),
                                                                                                          None)

        else:
            action_p1, action_p2 = self.generator.act(state, reward, round_num)

        action_p1 = random.choice(p1_available_actions) if action_p1 is None else action_p1
        action_p2 = random.choice(p2_available_actions) if action_p2 is None else action_p2

        return action_p1, action_p2


class FolkEgalAgent(Agent):
    def __init__(self, name: str, mdp_n_iterations: int, n_iterations: int, initial_state: GameState, file_name,
                 explore_prob=0.2, gamma=0.95, decay=0.9999954, read_from_file=False, specific_policy=False,
                 p1_weight=None, player=None):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.mdp_n_iterations = mdp_n_iterations
        self.n_iterations = n_iterations
        self.initial_state = initial_state
        self.explore_prob = explore_prob
        self.gamma = gamma
        self.decay = decay
        self.minimax_agent = MinimaxAgent('FolkEgalHelper', 0, initial_state, file_name, read_from_file=True) if \
            not specific_policy else None
        self.p1_policy_1 = {}
        self.p2_policy_1 = {}
        self.p1_policy_2 = {}
        self.p2_policy_2 = {}
        self.alternate_prob = -1.0
        self.specific_policy = specific_policy
        self.player = player

        if read_from_file:
            self._read_from_file(file_name)

        else:
            if specific_policy:
                assert p1_weight is not None and player is not None
                self._find_specific_policies(initial_state, p1_weight)

            else:
                self._find_policies(initial_state)

            with open('../agents/files/folk_egal/' + str(file_name) + '_p1_policy_1.json', 'w') as fp:
                json.dump(self.p1_policy_1, fp)

            with open('../agents/files/folk_egal/' + str(file_name) + '_p2_policy_1.json', 'w') as fp:
                json.dump(self.p2_policy_1, fp)

            with open('../agents/files/folk_egal/' + str(file_name) + '_p1_policy_2.json', 'w') as fp:
                json.dump(self.p1_policy_2, fp)

            with open('../agents/files/folk_egal/' + str(file_name) + '_p2_policy_2.json', 'w') as fp:
                json.dump(self.p2_policy_2, fp)

            with open('../agents/files/folk_egal/' + str(file_name) + '_alternate_prob.json', 'w') as fp:
                json.dump(self.alternate_prob, fp)

        self.p1_policy_to_use = self.p1_policy_1
        self.p2_policy_to_use = self.p2_policy_1

        if self.alternate_prob != -1:
            self._get_alternations()

    def _get_alternations(self):
        if self.alternate_prob == 0:
            self.alternations = [True] * 100
            return

        self.alternations = [0] * 100

        every_n_rounds = round(1 / self.alternate_prob)

        for i in range(len(self.alternations)):
            if (i + 1) % every_n_rounds == 0:
                self.alternations[i] = True

            else:
                self.alternations[i] = False

    def _read_from_file(self, file_name):
        with open('../agents/files/folk_egal/' + str(file_name) + '_p1_policy_1.json', 'r') as fp:
            self.p1_policy_1 = json.load(fp)

        with open('../agents/files/folk_egal/' + str(file_name) + '_p2_policy_1.json', 'r') as fp:
            self.p2_policy_1 = json.load(fp)

        with open('../agents/files/folk_egal/' + str(file_name) + '_p1_policy_2.json', 'r') as fp:
            self.p1_policy_2 = json.load(fp)

        with open('../agents/files/folk_egal/' + str(file_name) + '_p2_policy_2.json', 'r') as fp:
            self.p2_policy_2 = json.load(fp)

        with open('../agents/files/folk_egal/' + str(file_name) + '_alternate_prob.json', 'r') as fp:
            self.alternate_prob = json.load(fp)

    def act(self, state, reward, round_num):
        p1_available_actions, p2_available_actions = state.get_available_actions()

        if self.specific_policy:
            if self.player == P1:
                self.p1_policy_to_use, self.p2_policy_to_use = self.p1_policy_1, self.p2_policy_1

            else:
                self.p1_policy_to_use, self.p2_policy_to_use = self.p1_policy_2, self.p2_policy_2

            action_p1, action_p2 = self.p1_policy_to_use.get(str(state), None), self.p2_policy_to_use.get(str(state),
                                                                                                          None)

            action_p1 = random.choice(p1_available_actions) if action_p1 is None else action_p1
            action_p2 = random.choice(p2_available_actions) if action_p2 is None else action_p2

            return action_p1, action_p2

        if self.alternate_prob != -1.0:
            if round_num >= len(self.alternations):
                round_num -= len(self.alternations)

            alternate = self.alternations[round_num]

            if alternate:
                self.p1_policy_to_use, self.p2_policy_to_use = self.p1_policy_1, self.p2_policy_1

            else:
                self.p1_policy_to_use, self.p2_policy_to_use = self.p1_policy_2, self.p2_policy_2

        action_p1, action_p2 = self.p1_policy_to_use.get(str(state), None), self.p2_policy_to_use.get(str(state), None)

        action_p1 = random.choice(p1_available_actions) if action_p1 is None else action_p1
        action_p2 = random.choice(p2_available_actions) if action_p2 is None else action_p2

        return action_p1, action_p2

    def _find_specific_policies(self, initial_state: GameState, p1_weight: float):
        _, _, self.p1_policy_1, self.p2_policy_1 = self._solve_mdp(initial_state, p1_weight)
        _, _, self.p1_policy_2, self.p2_policy_2 = self._solve_mdp(initial_state, 1 - p1_weight)

    def _find_policies(self, initial_state: GameState):
        v1, v2 = self.minimax_agent.v_p1[str(initial_state)], self.minimax_agent.v_p2[str(initial_state)]

        R_p1, R_p2, pi_2_p1, pi_2_p2 = self._solve_mdp(initial_state, 1.0)
        L_p1, L_p2, pi_1_p1, pi_1_p2 = self._solve_mdp(initial_state, 0.0)

        if v1 < R_p1 < R_p2 and v2 < R_p2:
            self.p1_policy_1, self.p2_policy_1 = pi_2_p1, pi_2_p2

        elif L_p1 > L_p2 > v2 and L_p1 > v1:
            self.p1_policy_1, self.p2_policy_1 = pi_1_p1, pi_1_p2

        else:
            self.p1_policy_1, self.p2_policy_1, self.p1_policy_2, self.p2_policy_2 = pi_1_p1, pi_1_p2, pi_2_p1, pi_2_p2
            self._egal_search(R_p1, R_p2, L_p1, L_p2, initial_state)

    def _solve_mdp(self, initial_state: GameState, w: float):
        q_vals_p1 = {}
        v_p1 = {}
        p1_mdp_policy = {}
        alpha_p1 = 1.0
        q_vals_p2 = {}
        v_p2 = {}
        alpha_p2 = 1.0
        p2_mdp_policy = {}

        def find_reward(curr_state: GameState):
            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            p1_available_actions, p2_available_actions = curr_state.get_available_actions()
            q_val_action_pairs_p1 = [(q_vals_p1.get((str(curr_state), action), 0.0), action) for action in
                                     p1_available_actions]
            q_val_action_pairs_p2 = [(q_vals_p2.get((str(curr_state), action), 0.0), action) for action in
                                     p2_available_actions]

            action_p1 = max(q_val_action_pairs_p1, key=lambda x: x[0])[1]
            action_p2 = max(q_val_action_pairs_p2, key=lambda x: x[0])[1]

            next_state = curr_state.next(action_p1, action_p2)

            return find_reward(next_state)

        def rec_search(curr_state: GameState):
            nonlocal alpha_p1
            nonlocal alpha_p2

            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            explore_p1 = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])
            explore_p2 = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

            p1_available_actions, p2_available_actions = curr_state.get_available_actions()
            q_val_action_pairs_p1 = [(q_vals_p1.get((str(curr_state), action), 0.0), action) for action in
                                     p1_available_actions]
            q_val_action_pairs_p2 = [(q_vals_p2.get((str(curr_state), action), 0.0), action) for action in
                                     p2_available_actions]

            if explore_p1:
                action_p1 = random.choice(p1_available_actions)

            else:
                action_p1 = max(q_val_action_pairs_p1, key=lambda x: x[0])[1]

            if explore_p2:
                action_p2 = random.choice(p2_available_actions)

            else:
                action_p2 = max(q_val_action_pairs_p2, key=lambda x: x[0])[1]

            next_state = curr_state.next(action_p1, action_p2)

            reward_p1, reward_p2 = rec_search(next_state)
            if reward_p1 is None or reward_p2 is None or w is None:
                print(reward_p1)
                print(reward_p2)
                print(w)
            reward = reward_p1 * w + reward_p2 * (1 - w)

            # PLAYER 1
            q_vals_p1[(str(curr_state), action_p1)] = (1 - alpha_p1) * q_vals_p1.get((str(curr_state), action_p1),
                                                                                     0.0) + alpha_p1 * (
                                                              reward + self.gamma * v_p1.get(str(next_state), 0.0))

            q_val_action_pairs_p1 = [(q_vals_p1.get((str(curr_state), action), 0.0), action) for action in
                                     p1_available_actions]
            v_p1[str(curr_state)] = max(q_val_action_pairs_p1, key=lambda x: x[0])[0]

            p1_mdp_policy[str(curr_state)] = max(q_val_action_pairs_p1, key=lambda x: x[0])[1]

            # PLAYER 2
            q_vals_p2[(str(curr_state), action_p2)] = (1 - alpha_p2) * q_vals_p2.get((str(curr_state), action_p2), 0.0) \
                                                      + alpha_p2 * (
                                                              reward + self.gamma * v_p2.get(str(next_state), 0.0))

            q_val_action_pairs_p2 = [(q_vals_p2.get((str(curr_state), action), 0.0), action) for action in
                                     p2_available_actions]
            v_p2[str(curr_state)] = max(q_val_action_pairs_p2, key=lambda x: x[0])[0]

            p2_mdp_policy[str(curr_state)] = max(q_val_action_pairs_p2, key=lambda x: x[0])[1]

            alpha_p1 *= self.decay
            alpha_p2 *= self.decay

            return reward_p1, reward_p2

        for _ in range(self.mdp_n_iterations):
            rec_search(initial_state)

        r_p1, r_p2 = find_reward(initial_state)

        return r_p1, r_p2, p1_mdp_policy, p2_mdp_policy

    def _egal_search(self, L_p1, L_p2, R_p1, R_p2, initial_state: GameState):
        v1, v2 = self.minimax_agent.v_p1[str(initial_state)], self.minimax_agent.v_p2[str(initial_state)]

        def intersect(L_p1, L_p2, R_p1, R_p2):
            w = pulp.LpVariable("w", lowBound=0.0, upBound=1.0)
            model = LpProblem("Weight", LpMaximize)
            model += lpSum(np.array([L_p1 * w + L_p2 * w]))
            model += lpSum(np.array([L_p1 * w + L_p2 * w])) == lpSum(np.array([R_p1 * (1 - w) + R_p2 * (1 - w)]))

            model.solve(PULP_CBC_CMD(msg=False))

            for v in model.variables():
                try:
                    if 'w' in v.name:
                        self.alternate_prob = v.value()

                except:
                    print("Could not find LP variable value")

        def balance(L_p1, L_p2, R_p1, R_p2):
            w = pulp.LpVariable("w", lowBound=0.0, upBound=1.0)
            model = LpProblem("Weight", LpMaximize)
            model += lpSum(np.array([L_p1 * w + L_p2 * (1 - w)]))
            model += lpSum(np.array([L_p1 * w + L_p2 * (1 - w)])) == lpSum(np.array([R_p1 * w + R_p2 * (1 - w)]))

            model.solve(PULP_CBC_CMD(msg=False))

            for v in model.variables():
                try:
                    if 'w' in v.name:
                        return v.value()

                except:
                    print("Could not find LP variable value")

            return 0.5

        def rec_search(L_p1, L_p2, R_p1, R_p2, T):
            if T == 0:
                return intersect(L_p1, L_p2, R_p1, R_p2)

            w = balance(L_p1, L_p2, R_p1, R_p2)
            if w is None:
                print(f'Here: {w}')

            p1, p2, pi_p1, pi_p2 = self._solve_mdp(initial_state, w)

            if p1 * w + p2 * (1 - w) == L_p1 * w + L_p2 * (1 - w):
                return intersect(L_p1, L_p2, R_p1, R_p2)

            if v1 < p1 < p2 and v2 < p2:
                self.p1_policy_1, self.p2_policy_1 = pi_p1, pi_p2
                return rec_search(p1, p2, R_p1, R_p2, T - 1)

            else:
                self.p1_policy_2, self.p2_policy_2 = pi_p1, pi_p2
                return rec_search(L_p1, L_p2, p1, p2, T - 1)

        return rec_search(L_p1, L_p2, R_p1, R_p2, self.n_iterations)