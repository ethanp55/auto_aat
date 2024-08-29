import random
from games.prisoners_dilemma import PrisonersDilemma, baselines
from agents.folk_egal import FolkEgalAgent, FolkEgalPunishAgent
from agents.minimax_q import MinimaxAgent
from agents.alegaatr import ESTIMATES_LOOKBACK, TRAIN_RANDOM_PROB, TRAIN_RANDOM_N_ROUNDS_RATIOS, \
    distance_func, AssumptionChecker, Assumptions
from agents.cfr import CFRAgent
from utils.utils import P1, P2
import numpy as np
from copy import deepcopy
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collections import deque
import time

prisoner_game = PrisonersDilemma()
data_dir = '../aat/training_data/' + str(prisoner_game) + '/'


# Create opponent agents
def create_opponent_agents(player_idx):
    initial_state = prisoner_game.get_init_state()
    game_name = str(prisoner_game)

    # Experts in AlegAATr's pool
    coop_agent = FolkEgalAgent('CoopOpp', 1, 1, initial_state, game_name, read_from_file=True, player=player_idx)
    coop_punish_agent = FolkEgalPunishAgent('CoopPunishOpp', coop_agent, game_name, prisoner_game)
    bully_agent = FolkEgalAgent('BullyOpp', 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                                specific_policy=True, p1_weight=1.0, player=player_idx)
    bully_punish_agent = FolkEgalPunishAgent('BullyPunishOpp', bully_agent, game_name, prisoner_game)
    bullied_agent = FolkEgalAgent('BulliedOpp', 1, 1, initial_state,
                                  game_name + '_bullied', read_from_file=True, specific_policy=True,
                                  p1_weight=0.2, player=player_idx)
    minimax_agent = MinimaxAgent('MinimaxOpp', 1, initial_state, game_name, read_from_file=True)
    cfr_agent = CFRAgent(name='CfrOpp', initial_game_state=initial_state, n_iterations=1,
                         file_name=game_name, read_from_file=True)

    opponents = {coop_agent.name: coop_agent, coop_punish_agent.name: coop_punish_agent, bully_agent.name:
        bully_agent, bully_punish_agent.name: bully_punish_agent, bullied_agent.name: bullied_agent,
                 minimax_agent.name: minimax_agent, cfr_agent.name: cfr_agent}

    return opponents


n_epochs = 50
min_rounds = 50
max_rounds = 100
possible_rounds = list(range(min_rounds, max_rounds + 1))
training_data = {}

for use_random_switching in [True, False]:
    for epoch in range(1, n_epochs + 1):
        print('Epoch: ' + str(epoch))
        algaater_idx = np.random.choice([P1, P2])
        opponent_idx = 1 - algaater_idx

        print(algaater_idx)
        print(opponent_idx)

        opponents = create_opponent_agents(opponent_idx)
        assumption_checker = AssumptionChecker(prisoner_game, algaater_idx, baselines)
        experts = assumption_checker.experts

        # n_rounds = np.random.choice(possible_rounds)
        n_rounds = min_rounds

        for expert_key in experts.keys():
            expert_agent = deepcopy(experts[expert_key])
            expert_training_data = []

            for opponent_key in opponents.keys():
                opp_name = 'opp'
                playing_random, random_round_cnt, random_n_rounds = False, 0, 0
                opponent_agent = deepcopy(opponents[opponent_key])
                current_training_data = []
                reward_map = {opp_name: 0, expert_key: 0}
                prev_rewards = deque(maxlen=ESTIMATES_LOOKBACK)
                prev_opp_rewards = deque(maxlen=ESTIMATES_LOOKBACK)
                prev_assumptions = Assumptions(0, 0, 0, 0, 0, 0, 0)
                prev_i, prev_e, prev_v, prev_f, prev_b, prev_p, prev_u = deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK), \
                                                                         deque(maxlen=ESTIMATES_LOOKBACK)

                prev_reward_1 = 0
                prev_reward_2 = 0

                for round_num in range(n_rounds):
                    if playing_random:
                        random_round_cnt += 1

                    if playing_random and random_round_cnt > random_n_rounds:
                        playing_random, random_round_cnt, random_n_rounds = False, 0, 0

                    if not playing_random and use_random_switching:
                        playing_random = np.random.choice([1, 0], p=[TRAIN_RANDOM_PROB, 1 - TRAIN_RANDOM_PROB])

                        if playing_random:
                            random_n_rounds = int(random.choice(TRAIN_RANDOM_N_ROUNDS_RATIOS) * n_rounds)
                            opponent_agent = deepcopy(random.choice(list(opponents.values())))

                        else:
                            opponent_agent = deepcopy(opponents[opponent_key])

                    prisoner_game.reset()
                    state = deepcopy(prisoner_game.get_init_state())
                    action_map = dict()
                    opp_actions = []

                    key_agent_map = {expert_key: expert_agent, opp_name: opponent_agent} if algaater_idx == P1 else \
                        {opp_name: opponent_agent, expert_key: expert_agent}

                    rewards_1 = []
                    rewards_2 = []

                    while not state.is_terminal():
                        for agent_key, agent in key_agent_map.items():
                            agent_reward = prev_reward_1 if agent_key == expert_key else prev_reward_2
                            agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                            action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                            if agent_key == opp_name and state.turn == opponent_idx:
                                opp_action = agent_action1 if opponent_agent.player == P1 else agent_action2
                                opp_actions.append(opp_action)

                            # elif agent_key != opponent_key:
                            #     assumption_checker.act(state, agent_reward, round_num)

                        updated_rewards_map, next_state = prisoner_game.execute_agent_action(action_map)

                        for agent_name, new_reward in updated_rewards_map.items():
                            reward_map[agent_name] += new_reward

                            if agent_name == expert_key:
                                rewards_1.append(new_reward)

                            else:
                                rewards_2.append(new_reward)

                        prev_state = deepcopy(state)
                        state = next_state

                    prev_reward_1 = sum(rewards_1)
                    prev_reward_2 = sum(rewards_2)
                    prev_rewards.append(prev_reward_1)
                    prev_opp_rewards.append(prev_reward_2)
                    agent_reward = reward_map[expert_key]
                    proposed_avg_payoff = baselines[expert_key]
                    n_remaining_rounds = n_rounds - round_num - 1
                    proposed_payoff_to_go = proposed_avg_payoff * n_remaining_rounds
                    proposed_total_payoff = agent_reward + proposed_payoff_to_go
                    # short_term, medium_term, long_term = assumption_checker.estimate_assumptions(prev_short_term,
                    #                                                                              prev_medium_term,
                    #                                                                              prev_long_term,
                    #                                                                              prev_rewards,
                    #                                                                              prev_opp_rewards,
                    #                                                                              round_num, expert_agent)
                    new_assumptions = assumption_checker.estimate_assumptions(prev_rewards, prev_opp_rewards, round_num,
                                                                              expert_agent)

                    prev_i.append(new_assumptions.improvement_assumption)
                    prev_e.append(new_assumptions.efficient_assumption)
                    prev_v.append(new_assumptions.vengeful_assumption)
                    prev_f.append(new_assumptions.fair_assumption)
                    prev_b.append(new_assumptions.bully_assumption)
                    prev_p.append(new_assumptions.pushover_assumption)
                    prev_u.append(new_assumptions.understands_me_assumption)

                    prev_assumptions = Assumptions(sum(prev_i) / len(prev_i), sum(prev_e) / len(prev_e),
                                                   sum(prev_v) / len(prev_v), sum(prev_f) / len(prev_f),
                                                   sum(prev_b) / len(prev_b), sum(prev_p) / len(prev_p),
                                                   sum(prev_u) / len(prev_u))

                    curr_tup = [round_num, (agent_reward / (round_num + 1)), prev_assumptions.improvement_assumption,
                                prev_assumptions.efficient_assumption, prev_assumptions.vengeful_assumption,
                                prev_assumptions.fair_assumption, prev_assumptions.bully_assumption,
                                prev_assumptions.pushover_assumption, prev_assumptions.understands_me_assumption,
                                algaater_idx, proposed_avg_payoff, proposed_total_payoff]

                    current_training_data.append(curr_tup)

                total_payoff = reward_map[expert_key]

                for tup in current_training_data:
                    tup[-1] = total_payoff / n_rounds
                    tup[-2] = (total_payoff / n_rounds) / tup[-2] if tup[-2] != 0 else (total_payoff / n_rounds) / 0.000001

                expert_training_data.extend(current_training_data)

            training_data[expert_key] = training_data.get(expert_key, []) + expert_training_data

for expert_key, data in training_data.items():
    with open(data_dir + expert_key + '_training_data.pickle', 'wb') as f:
        pickle.dump(data, f)

time.sleep(5)

print('Training KNN model...')

experts = AssumptionChecker(prisoner_game, P1, baselines).experts

for expert_key, _ in experts.items():
    training_data_file = expert_key + '_training_data.pickle'

    with open(data_dir + training_data_file, 'rb') as f:
        training_data = np.array(pickle.load(f))

    x = training_data[:, 0:-2]
    y = training_data[:, -1]

    print('X train shape: ' + str(x.shape))
    print('Y train shape: ' + str(y.shape))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = NearestNeighbors(n_neighbors=15, metric=distance_func)
    model.fit(x_scaled)

    trained_knn_file = expert_key + '_trained_knn_aat.pickle'
    trained_knn_scaler_file = expert_key + '_trained_knn_scaler_aat.pickle'

    with open(data_dir + trained_knn_file, 'wb') as f:
        pickle.dump(model, f)

    with open(data_dir + trained_knn_scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
