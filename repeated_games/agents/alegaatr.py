from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from repeated_games.agents.folk_egal import FolkEgalAgent, FolkEgalPunishAgent
from repeated_games.agents.minimax_q import MinimaxAgent
from repeated_games.agents.cfr import CFRAgent
from utils.utils import P1, P2, PAD_VAL, NETWORK_NAME
import numpy as np
from typing import Tuple, List
import pickle
from copy import deepcopy
import random
from sklearn.neighbors import KernelDensity
from tensorflow.keras.models import load_model
from collections import deque

ESTIMATES_LOOKBACK = 25
N_SIMULATIONS = 10
TRAIN_RANDOM_PROB, TRAIN_RANDOM_N_ROUNDS_RATIOS = 0.20, [0.025, 0.05, 0.10, 0.15, 0.20]


class Assumptions:
    def __init__(self, improvement_assumption, efficient_assumption, vengeful_assumption, fair_assumption,
                 bully_assumption, pushover_assumption, understands_me_assumption):
        self.improvement_assumption = improvement_assumption
        self.efficient_assumption = efficient_assumption
        self.vengeful_assumption = vengeful_assumption
        self.fair_assumption = fair_assumption
        self.bully_assumption = bully_assumption
        self.pushover_assumption = pushover_assumption
        self.understands_me_assumption = understands_me_assumption


class AssumptionChecker:
    def __init__(self, game: MarkovGameMDP, player: int, baselines):
        self.player = player
        self.experts = AlegAATr.create_aat_experts(game, player)
        self.attack_actions = []
        self.coop_actions = []
        self.bully_actions = []
        self.bullied_actions = []
        self.cfr_actions = []
        self.game = deepcopy(game)

        initial_state = self.game.get_init_state()

        self.game_is_simultaneous = initial_state.is_simultaneous()

        self.pareto_reward = baselines['AlgaaterCoop']

    def act(self, state, reward, round_num):
        minimax_agent = self.experts['AlgaaterMinimax']
        p1_attack_policy, p2_attack_policy = minimax_agent.p1_attack_policy, minimax_agent.p2_attack_policy
        attack_action_p1, attack_action_p2 = p1_attack_policy.get(str(state), None), p2_attack_policy.get(str(state),
                                                                                                          None)
        attack_action = attack_action_p1 if self.player == P2 else attack_action_p2

        coop_agent = self.experts['AlgaaterCoop']
        coop_action_p1, coop_action_p2 = coop_agent.act(state, reward, round_num)
        coop_action = coop_action_p1 if self.player == P2 else coop_action_p2

        bully_agent = self.experts['AlgaaterBully']
        bully_action_p1, bully_action_p2 = bully_agent.act(state, reward, round_num)
        bully_action = bully_action_p1 if self.player == P2 else bully_action_p2

        bullied_agent = self.experts['AlgaaterBullied']
        bullied_action_p1, bullied_action_p2 = bullied_agent.act(state, reward, round_num)
        bullied_action = bullied_action_p1 if self.player == P2 else bullied_action_p2

        cfr_agent = self.experts['AlgaaterCfr']
        cfr_action_p1, cfr_action_p2 = cfr_agent.act(state, reward, round_num)
        cfr_action = cfr_action_p1 if self.player == P2 else cfr_action_p2

        if self.game_is_simultaneous or state.turn != self.player:
            self.attack_actions.append(attack_action)
            self.coop_actions.append(coop_action)
            self.bully_actions.append(bully_action)
            self.bullied_actions.append(bullied_action)
            self.cfr_actions.append(cfr_action)

    def estimate_assumptions(self, prev_rewards, prev_opp_rewards, round_num: int, curr_agent: Agent) -> Assumptions:
        if 'Punish' in curr_agent.name:
            agent_copy = deepcopy(curr_agent)

        else:
            agent_copy = curr_agent

        improvement_assumption = self._improvement_assumption_checker(prev_rewards)
        efficient_assumption = self._efficient_assumption_checker(prev_rewards)
        vengeful_assumption = self._vengeful_assumption_checker(prev_rewards[-1], agent_copy, round_num)
        fair_assumption = self._fair_assumption_checker(prev_rewards[-1], agent_copy, round_num)
        bully_assumption = self._bully_assumption_checker(prev_rewards[-1], agent_copy, round_num)
        pushover_assumption = self._pushover_assumption_checker(prev_rewards[-1], agent_copy, round_num)
        understands_me_assumption = self._understands_me_assumption_checker(agent_copy, vengeful_assumption,
                                                                            bully_assumption, fair_assumption,
                                                                            pushover_assumption)

        new_assumptions = Assumptions(improvement_assumption, efficient_assumption, vengeful_assumption,
                                      fair_assumption, bully_assumption, pushover_assumption, understands_me_assumption)

        return new_assumptions

    def _improvement_assumption_checker(self, prev_rewards):
        for i in range(1, len(prev_rewards)):
            if prev_rewards[i] < prev_rewards[i - 1]:
                return int(False)

        return int(True)

    def _efficient_assumption_checker(self, prev_rewards):
        if len(prev_rewards) == 0:
            return 1.0

        gte_pareto_vals = [int(reward >= self.pareto_reward) for reward in prev_rewards]

        return np.array(gte_pareto_vals).mean()

    def _vengeful_assumption_checker(self, reward: float, curr_agent: Agent, round_num):
        rewards_1 = []
        rewards_2 = []
        minimax_agent = self.experts['AlgaaterMinimax']
        p1_attack_policy, p2_attack_policy = minimax_agent.p1_attack_policy, minimax_agent.p2_attack_policy

        for _ in range(N_SIMULATIONS):
            self.game.reset()
            curr_state = deepcopy(self.game.get_init_state())
            reward_1, reward_2 = 0, 0
            action_map = dict()

            while not curr_state.is_terminal():
                p1_available_actions, p2_available_actions = curr_state.get_available_actions()
                action_p1, action_p2 = curr_agent.act(curr_state, 0, round_num)
                attack_action_p1, attack_action_p2 = p1_attack_policy.get(str(curr_state), None), p2_attack_policy.get(
                    str(curr_state),
                    None)

                attack_action_p1 = random.choice(p1_available_actions) if attack_action_p1 is None else attack_action_p1
                attack_action_p2 = random.choice(p2_available_actions) if attack_action_p2 is None else attack_action_p2

                if self.player == P1:
                    action_map['1'], action_map['2'] = action_p1, attack_action_p2

                else:
                    action_map['1'], action_map['2'] = attack_action_p1, action_p2

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

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(rewards).reshape(-1, 1))
        log_likelihood = kde.score_samples(np.array(reward).reshape(1, -1))[0]

        return log_likelihood

    def _fair_assumption_checker(self, reward: float, curr_agent: Agent, round_num):
        rewards_1 = []
        rewards_2 = []
        coop_agent = self.experts['AlgaaterCoop']
        prev_player = coop_agent.player
        coop_agent.player = 1 - self.player

        for _ in range(N_SIMULATIONS):
            self.game.reset()
            curr_state = deepcopy(self.game.get_init_state())
            reward_1, reward_2 = 0, 0
            action_map = dict()

            while not curr_state.is_terminal():
                action_p1, action_p2 = curr_agent.act(curr_state, 0, round_num)
                coop_action_p1, coop_action_p2 = coop_agent.act(curr_state, 0, round_num)

                if self.player == P1:
                    action_map['1'], action_map['2'] = action_p1, coop_action_p2

                else:
                    action_map['1'], action_map['2'] = coop_action_p1, action_p2

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

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(rewards).reshape(-1, 1))
        log_likelihood = kde.score_samples(np.array(reward).reshape(1, -1))[0]

        coop_agent.player = prev_player

        return log_likelihood

    def _bully_assumption_checker(self, reward: float, curr_agent: Agent, round_num):
        rewards_1 = []
        rewards_2 = []
        bully_agent = self.experts['AlgaaterBully']
        prev_player = bully_agent.player
        bully_agent.player = 1 - self.player

        for _ in range(N_SIMULATIONS):
            self.game.reset()
            curr_state = deepcopy(self.game.get_init_state())
            reward_1, reward_2 = 0, 0
            action_map = dict()

            while not curr_state.is_terminal():
                action_p1, action_p2 = curr_agent.act(curr_state, 0, round_num)
                bully_action_p1, bully_action_p2 = bully_agent.act(curr_state, 0, round_num)

                if self.player == P1:
                    action_map['1'], action_map['2'] = action_p1, bully_action_p2

                else:
                    action_map['1'], action_map['2'] = bully_action_p1, action_p2

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
        # min_reward, max_reward = min(rewards), max(rewards)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(rewards).reshape(-1, 1))
        log_likelihood = kde.score_samples(np.array(reward).reshape(1, -1))[0]

        bully_agent.player = prev_player

        return log_likelihood

    def _pushover_assumption_checker(self, reward: float, curr_agent: Agent, round_num):
        rewards_1 = []
        rewards_2 = []
        bullied_agent = self.experts['AlgaaterBullied']
        prev_player = bullied_agent.player
        bullied_agent.player = 1 - self.player

        for _ in range(N_SIMULATIONS):
            self.game.reset()
            curr_state = deepcopy(self.game.get_init_state())
            reward_1, reward_2 = 0, 0
            action_map = dict()

            while not curr_state.is_terminal():
                action_p1, action_p2 = curr_agent.act(curr_state, 0, round_num)
                bully_action_p1, bully_action_p2 = bullied_agent.act(curr_state, 0, round_num)

                if self.player == P1:
                    action_map['1'], action_map['2'] = action_p1, bully_action_p2

                else:
                    action_map['1'], action_map['2'] = bully_action_p1, action_p2

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
        # min_reward, max_reward = min(rewards), max(rewards)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(rewards).reshape(-1, 1))
        log_likelihood = kde.score_samples(np.array(reward).reshape(1, -1))[0]

        bullied_agent.player = prev_player

        return log_likelihood

    def _understands_me_assumption_checker(self, curr_agent: Agent, vengeful_estimate, bully_estimate, fair_estimate,
                                           pushover_assumption):
        curr_agent_name = curr_agent.name

        if 'Bully' in curr_agent_name and 'Punish' in curr_agent_name:
            return pushover_assumption

        elif 'Bully' in curr_agent_name:
            return vengeful_estimate

        elif 'Bullied' in curr_agent_name or 'Minimax' in curr_agent_name or \
                ('Coop' in curr_agent_name and 'Punish' not in curr_agent_name):
            return bully_estimate

        elif 'CoopPunish' in curr_agent_name:
            return fair_estimate

        elif 'Cfr' in curr_agent_name:
            return int(False)

        else:
            raise Exception('Invalid agent in use: ' + str(curr_agent_name))


def distance_func(x, y):
    round_num_dist = 2 * abs(x[0] - y[0])
    curr_avg_payoff_dist = 2 * abs(x[1] - y[1])

    improvement_assumption_dist = 4 * abs(x[2] - y[2])

    efficient_assumption_dist = 4 * abs(x[3] - y[3])

    vengeful_assumption_dist = 4 * abs(x[4] - y[4])

    fair_assumption_dist = 4 * abs(x[5] - y[5])

    bully_assumption_dist = 4 * abs(x[6] - y[6])

    pushover_assumption_dist = 4 * abs(x[7] - y[7])

    understands_me_assumption_dist = 4 * abs(x[8] - y[8])

    player_dist = 1 * abs(x[9] - y[9])

    return sum(
        [round_num_dist, curr_avg_payoff_dist, improvement_assumption_dist, efficient_assumption_dist,
         vengeful_assumption_dist, fair_assumption_dist, bully_assumption_dist, pushover_assumption_dist,
         understands_me_assumption_dist, player_dist])


class AlegAATr(Agent):
    def __init__(self, name: str, game: MarkovGameMDP, player: int, baselines, use_nn=False, lmbda=0.95, n_simulations=10, lookback=3, log=False, use_auto_aat=False):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.game = deepcopy(game)
        self.player = player
        self.baselines = baselines
        self.use_nn = use_nn
        self.lmbda = lmbda
        self.n_simulations = n_simulations
        self.lookback = lookback
        self.log = log
        self.log_message = ''
        self.assumption_checker = AssumptionChecker(self.game, player, baselines)
        self.experts = self.assumption_checker.experts
        self.expert_to_use = self.experts['AlgaaterCoop']
        self.use_auto_aat = use_auto_aat
        self.models, self.scalers, self.training_datas = self._load_model_data(self.game)
        self._get_best_self_play_expert(self.game)
        self.curr_expert_reward = 0.0
        self.curr_expert_n_rounds = 0
        self.switch_round_num = 0
        self.recent_rewards = []
        self.recent_opp_rewards = []
        self.agent_averages = {}
        self.corrections = {}
        self.n_rounds_since_played = {}

        for expert_key in self.experts.keys():
            self.corrections[expert_key] = deque(maxlen=self.lookback)
            self.n_rounds_since_played[expert_key] = 0

        self.prev_predictions = {}

        self.minimax_val = self.experts['AlgaaterMinimax'].v_p1[str(self.game.get_init_state())] if self.player == P1 \
            else self.experts['AlgaaterMinimax'].v_p2[str(self.game.get_init_state())]

        self.prev_i, self.prev_e, self.prev_v, self.prev_f, self.prev_b, self.prev_p, self.prev_u = \
            deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), \
            deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), \
            deque(maxlen=ESTIMATES_LOOKBACK)

        if self.use_auto_aat:
            self.assumption_pred_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
            self.state_scaler = pickle.load(open(f'../../networks/scalers/{NETWORK_NAME}_state_scaler.pickle', 'rb'))
            assert self.state_scaler._scaler is not None

    def _get_best_self_play_expert(self, game: MarkovGameMDP):
        self.best_self_play_expert = None
        self.best_self_play_p1_average = -np.inf
        self.best_self_play_p2_average = -np.inf

        for expert_name, expert in self.experts.items():
            rewards_1 = []
            rewards_2 = []

            agent1 = deepcopy(expert)
            agent1.name, agent1.player = 'Agent1', P1

            agent2 = deepcopy(expert)
            agent2.name, agent2.player = 'Agent2', P2

            for round_num in range(100):
                game.reset()
                curr_state = deepcopy(game.get_init_state())
                reward_1, reward_2 = 0, 0
                action_map = dict()

                while not curr_state.is_terminal():
                    action_p1, _ = agent1.act(curr_state, 0, round_num)
                    _, action_p2 = agent2.act(curr_state, 0, round_num)

                    action_map['1'], action_map['2'] = action_p1, action_p2

                    updated_rewards_map, next_curr_state = game.execute_agent_action(action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        if agent_name == '1':
                            reward_1 += new_reward

                        else:
                            reward_2 += new_reward

                    curr_state = next_curr_state

                rewards_1.append(reward_1)
                rewards_2.append(reward_2)

            p1_avg = np.array(rewards_1).mean()
            p2_avg = np.array(rewards_2).mean()

            if p1_avg > self.best_self_play_p1_average and p2_avg > self.best_self_play_p2_average:
                self.best_self_play_p1_average = p1_avg
                self.best_self_play_p2_average = p2_avg
                self.best_self_play_expert = expert_name

    def _load_model_data(self, game: MarkovGameMDP):
        models = {}
        scalers = {}
        training_datas = {}

        data_dir = '../aat/training_data/' + str(game) + '/'

        for e_key in self.experts.keys():
            if self.use_nn:
                models[e_key] = pickle.load(open(data_dir + e_key + '_trained_nn_aat.pickle', 'rb'))
                scalers[e_key] = pickle.load(open(data_dir + e_key + '_trained_nn_scaler_aat.pickle', 'rb'))

            elif self.use_auto_aat:
                models[e_key] = pickle.load(open(data_dir + e_key + '_trained_knn_aat_auto.pickle', 'rb'))
                scalers[e_key] = pickle.load(open(data_dir + e_key + '_trained_knn_scaler_aat_auto.pickle', 'rb'))
                training_datas[e_key] = np.array(pickle.load(open(data_dir + e_key + '_training_data_auto.pickle', 'rb')))

            else:
                models[e_key] = pickle.load(open(data_dir + e_key + '_trained_knn_aat.pickle', 'rb'))
                scalers[e_key] = pickle.load(open(data_dir + e_key + '_trained_knn_scaler_aat.pickle', 'rb'))
                training_datas[e_key] = np.array(pickle.load(open(data_dir + e_key + '_training_data.pickle', 'rb')))

        return models, scalers, training_datas

    def _knn_aat_prediction_func(self, x: List, expert_key: str) -> Tuple[List, List, List]:
        model = self.models[expert_key]
        scaler = self.scalers[expert_key]
        training_data = self.training_datas[expert_key]

        x = np.array(x).reshape(1, -1)
        x_scaled = scaler.transform(x)
        neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, 15)

        predictions = []
        corrections = []
        distances = []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            predictions.append(training_data[neighbor_idx, -1])
            corrections.append(training_data[neighbor_idx, -2])
            distances.append(neighbor_dist)

        return predictions, corrections, distances

    def update_expert(self, prev_rewards, prev_opp_rewards, round_num, proportion_payoff, proposed_total_payoff,
                      agent_reward, n_rounds, state, prev_reward_1, prev_reward_2, actions, description):
        if self.use_auto_aat:
            assert self.assumption_pred_model is not None
            assert self.state_scaler is not None

            # g_description = PRISONERS_G_DESCRIPTIONS[self.expert_to_use.name]
            g_description = 'Generator for playing the prisoner\'s dilemma game'

            state_input = [actions.index(state.actions[self.player]),
                           actions.index(state.actions[1 - self.player]),
                           prev_reward_1,
                           prev_reward_2]
            state_input += [PAD_VAL] * (300 - len(state_input))
            state_input_scaled = self.state_scaler.scale(np.array(state_input).reshape(1, -1))
            new_assumptions = self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                                                     np.array(description).reshape(1, -1),
                                                     state_input_scaled)).numpy()
            new_assumptions = new_assumptions[0, :7]

            self.prev_i.append(new_assumptions[0])
            self.prev_e.append(new_assumptions[1])
            self.prev_v.append(new_assumptions[2])
            self.prev_f.append(new_assumptions[3])
            self.prev_b.append(new_assumptions[4])
            self.prev_p.append(new_assumptions[5])
            self.prev_u.append(new_assumptions[6])

        else:
            new_assumptions = self.assumption_checker.estimate_assumptions(prev_rewards, prev_opp_rewards, round_num,
                                                                           self.expert_to_use)

            self.prev_i.append(new_assumptions.improvement_assumption)
            self.prev_e.append(new_assumptions.efficient_assumption)
            self.prev_v.append(new_assumptions.vengeful_assumption)
            self.prev_f.append(new_assumptions.fair_assumption)
            self.prev_b.append(new_assumptions.bully_assumption)
            self.prev_p.append(new_assumptions.pushover_assumption)
            self.prev_u.append(new_assumptions.understands_me_assumption)

        i_avg = sum(self.prev_i) / len(self.prev_i)
        e_avg = sum(self.prev_e) / len(self.prev_e)
        v_avg = sum(self.prev_v) / len(self.prev_v)
        f_avg = sum(self.prev_f) / len(self.prev_f)
        b_avg = sum(self.prev_b) / len(self.prev_b)
        p_avg = sum(self.prev_p) / len(self.prev_p)
        u_avg = sum(self.prev_u) / len(self.prev_u)

        self.curr_expert_reward += prev_rewards[-1]
        self.curr_expert_n_rounds += 1

        self.recent_rewards.append(prev_rewards[-1])
        self.recent_opp_rewards.append(prev_opp_rewards[-1])

        self.corrections[self.expert_to_use.name].append(prev_rewards[-1])

        if self.log:
            self.log_message = f'{round_num};{i_avg};{e_avg};{v_avg};{f_avg};{b_avg};{p_avg};{u_avg};'

        if round_num == 0:
            # For the first round just return and use the coop agent (initialized in the constructor)
            return new_assumptions

        if round_num > self.switch_round_num:
            tup = [round_num, proportion_payoff, i_avg, e_avg, v_avg, f_avg, b_avg, p_avg, u_avg, self.player,
                   proposed_total_payoff, proposed_total_payoff]

            res = {}

            for expert_key in self.experts.keys():
                if self.use_nn:
                    nn_model, nn_scaler = self.models[expert_key], self.scalers[expert_key]
                    formatted_tup = np.array(tup[:-2]).reshape(1, -1)
                    pred = nn_model.predict(nn_scaler.transform(formatted_tup))[0]
                    correction_term = (sum(self.corrections[expert_key]) / len(
                        self.corrections[expert_key])) / pred if len(
                        self.corrections[expert_key]) > 0 else 1.0
                    res[expert_key] = pred * correction_term
                    self.prev_predictions[expert_key] = pred

                    # print(f'{expert_key} -- {pred}')

                else:
                    predictions, corrections, distances = self._knn_aat_prediction_func(tup[:-2], expert_key)

                    total_payoff_pred = 0
                    inverse_distance_sum = 0

                    for dist in distances:
                        inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

                    for i in range(len(predictions)):
                        distance_i = distances[i]
                        cor = corrections[i]
                        inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
                        distance_weight = inverse_distance_i / inverse_distance_sum

                        total_payoff_pred += (self.baselines[expert_key] * cor * distance_weight)

                    if len(self.corrections[expert_key]) > 0:
                        self.n_rounds_since_played[expert_key] += 1 if expert_key != self.expert_to_use.name else 0
                        prob = self.lmbda ** self.n_rounds_since_played[expert_key]
                        use_empricial_avgs = np.random.choice([1, 0], p=[prob, 1 - prob])

                    else:
                        use_empricial_avgs = False

                    if self.log:
                        prob_for_log = 1.0 if len(self.corrections[expert_key]) == 0 else self.lmbda ** \
                                                                                          self.n_rounds_since_played[
                                                                                              expert_key]
                        self.log_message += f'{prob_for_log}-{use_empricial_avgs};'

                    res[expert_key] = total_payoff_pred if not use_empricial_avgs else sum(self.corrections[expert_key]) / len(self.corrections[expert_key])
                    self.prev_predictions[expert_key] = total_payoff_pred

                    if self.log:
                        emp_avg_log = sum(self.corrections[expert_key]) / len(self.corrections[expert_key]) if len(self.corrections[expert_key]) > 0 else np.nan
                        self.log_message += f'{total_payoff_pred}-{emp_avg_log};'

                    # print(f'{expert_key} -- {res[expert_key]}')

            expert_key = max(res, key=lambda key: res[key])
            best_key = expert_key
            self.n_rounds_since_played[best_key] = 0

            old_expert = self.expert_to_use.name
            self.expert_to_use = self.experts[best_key]

            # print(f'AlgAATer expert: {best_key}')

            if old_expert != self.expert_to_use.name and isinstance(self.expert_to_use, FolkEgalPunishAgent):
                self.expert_to_use.start_round, self.expert_to_use.should_attack = round_num + 1, False

            self.curr_expert_reward = 0.0
            self.curr_expert_n_rounds = 0
            n_rounds = 1
            self.switch_round_num = round_num + n_rounds
            self.recent_rewards.clear()
            self.recent_opp_rewards.clear()

            if self.log:
                self.log_message += f'{self.expert_to_use.name}'

        return new_assumptions

    def act(self, state, reward, round_num):
        return self.expert_to_use.act(state, reward, round_num)

    def reset_expert(self):
        for expert in self.experts.values():
            if isinstance(expert, FolkEgalPunishAgent):
                expert.start_round, expert.should_attack = 0, False

        self.expert_to_use = self.experts['AlgaaterCoop']
        self.curr_expert_reward = 0.0
        self.curr_expert_n_rounds = 0
        self.switch_round_num = 0
        self.recent_rewards.clear()
        self.recent_opp_rewards.clear()
        self.agent_averages = {}

        for expert_key in self.experts.keys():
            # self.corrections[expert_key] = 1.0
            self.corrections[expert_key] = deque(maxlen=self.lookback)
            self.n_rounds_since_played[expert_key] = 0

        self.prev_predictions = {}
        self.prev_i, self.prev_e, self.prev_v, self.prev_f, self.prev_b, self.prev_p, self.prev_u = \
            deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), \
            deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), deque(maxlen=ESTIMATES_LOOKBACK), \
            deque(maxlen=ESTIMATES_LOOKBACK)
        self.log_message = ''

    @staticmethod
    def create_aat_experts(game: MarkovGameMDP, player: int):
        initial_state = game.get_init_state()
        game_name = str(game)

        coop_agent = FolkEgalAgent('AlgaaterCoop', 1, 1, initial_state, game_name, read_from_file=True, player=player)
        coop_punish_agent = FolkEgalPunishAgent('AlgaaterCoopPunish', coop_agent, game_name, game)
        bully_agent = FolkEgalAgent('AlgaaterBully', 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                                    specific_policy=True, p1_weight=1.0 - player, player=player)
        bully_punish_agent = FolkEgalPunishAgent('AlgaaterBullyPunish', bully_agent, game_name, game)
        bullied_agent = FolkEgalAgent('AlgaaterBullied', 1, 1, initial_state,
                                      game_name + '_bullied', read_from_file=True, specific_policy=True,
                                      p1_weight=abs(0.2 - player), player=player)
        minimax_agent = MinimaxAgent('AlgaaterMinimax', 1, initial_state, game_name, read_from_file=True, player=player)
        cfr_agent = CFRAgent(name='AlgaaterCfr', initial_game_state=initial_state, n_iterations=1,
                             file_name=game_name, read_from_file=True, player=player)

        experts = {coop_agent.name: coop_agent, coop_punish_agent.name: coop_punish_agent, bully_agent.name:
            bully_agent, bully_punish_agent.name: bully_punish_agent, bullied_agent.name: bullied_agent,
                   minimax_agent.name: minimax_agent, cfr_agent.name: cfr_agent}

        return experts