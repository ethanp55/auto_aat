from repeated_games.chicken_game import ChickenGame, baselines as chicken_baselines, ACTIONS as chicken_actions
from repeated_games.coordination_game import CoordinationGame, baselines as coord_baselines, ACTIONS as coord_actions
from repeated_games.prisoners_dilemma import PrisonersDilemma, baselines as pd_baselines, ACTIONS as pd_actions
from repeated_games.agents.alegaatr import AlegAATr, ESTIMATES_LOOKBACK, Assumptions
from utils.utils import CHICKEN_E_DESCRIPTION, COORD_E_DESCRIPTION, P1, P2, PRISONERS_E_DESCRIPTION
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import csv

game = PrisonersDilemma()

baselines = pd_baselines if str(game) == 'prisoners_dilemma_game' else \
    (chicken_baselines if str(game) == 'chicken_game' else coord_baselines)
ACTIONS = pd_actions if str(game) == 'prisoners_dilemma_game' else \
    (chicken_actions if str(game) == 'chicken_game' else coord_actions)
DESCRIPTION = PRISONERS_E_DESCRIPTION if str(game) == 'prisoners_dilemma_game' else \
    (CHICKEN_E_DESCRIPTION if str(game) == 'chicken_game' else COORD_E_DESCRIPTION)

n_epochs = 50
min_rounds = 50
max_rounds = 100
possible_rounds = list(range(min_rounds, max_rounds + 1))
total_rewards_1 = []
total_rewards_2 = []
use_auto_aat = False
auto_aat_tuned = True

file_modifier = '_auto' if use_auto_aat else ('_auto_tuned' if auto_aat_tuned else '')
vector_file = f'../../analysis/{str(game)}_vectors/Alegaatr1_self_play{file_modifier}.csv'
with open(vector_file, 'w', newline='') as _:
    pass

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))

    epoch_rewards_1 = []
    epoch_rewards_2 = []

    algaater_idx_1 = np.random.choice([P1, P2])
    algaater_idx_2 = 1 - algaater_idx_1

    algaater_1 = AlegAATr('Algaater1', game, algaater_idx_1, baselines, use_auto_aat=use_auto_aat, auto_aat_tuned=auto_aat_tuned)
    algaater_2 = AlegAATr('Algaater2', game, algaater_idx_2, baselines, use_auto_aat=use_auto_aat, auto_aat_tuned=auto_aat_tuned)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    reward_map = {algaater_1.name: 0, algaater_2.name: 0}
    prev_rewards_1 = deque(maxlen=ESTIMATES_LOOKBACK)
    prev_rewards_2 = deque(maxlen=ESTIMATES_LOOKBACK)

    # prev_short_term_1 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_medium_term_1 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_long_term_1 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_short_term_2 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_medium_term_2 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_long_term_2 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    prev_assumptions_1 = Assumptions(0, 0, 0, 0, 0, 0, 0)
    prev_assumptions_2 = Assumptions(0, 0, 0, 0, 0, 0, 0)

    prev_reward_1 = 0
    prev_reward_2 = 0

    for round_num in range(n_rounds):
        game.reset()
        state = deepcopy(game.get_init_state())
        action_map = dict()
        opp_actions_1 = []
        opp_actions_2 = []

        key_agent_map = {algaater_1.name: algaater_1, algaater_2.name: algaater_2} if algaater_idx_1 == P1 else \
            {algaater_2.name: algaater_2, algaater_1.name: algaater_1}

        rewards_1 = []
        rewards_2 = []

        while not state.is_terminal():
            for agent_key, agent in key_agent_map.items():
                agent_reward = prev_reward_1 if agent_key == algaater_1.name else prev_reward_2
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                if agent_key == algaater_1.name:
                    # algaater_1.assumption_checker.act(state, agent_reward, round_num)

                    opp_action = agent_action1 if algaater_1.player == P1 else agent_action2
                    opp_actions_2.append(opp_action)

                else:
                    # algaater_2.assumption_checker.act(state, agent_reward, round_num)

                    opp_action = agent_action1 if algaater_2.player == P1 else agent_action2
                    opp_actions_1.append(opp_action)

            updated_rewards_map, next_state = game.execute_agent_action(action_map)

            for agent_name, new_reward in updated_rewards_map.items():
                reward_map[agent_name] += new_reward

                if agent_name == algaater_1.name:
                    rewards_1.append(new_reward)

                else:
                    rewards_2.append(new_reward)

            state = next_state

        prev_reward_1 = sum(rewards_1)
        prev_reward_2 = sum(rewards_2)
        prev_rewards_1.append(prev_reward_1)
        epoch_rewards_1.append(reward_map[algaater_1.name])
        prev_rewards_2.append(prev_reward_2)
        epoch_rewards_2.append(reward_map[algaater_2.name])
        proposed_avg_payoff = baselines[algaater_1.expert_to_use.name]
        n_remaining_rounds = n_rounds - round_num - 1
        proposed_payoff_to_go = proposed_avg_payoff * n_remaining_rounds

        for agent_key, agent in key_agent_map.items():
            prev_assumptions = prev_assumptions_1 if agent_key == algaater_1.name else prev_assumptions_2
            prev_rewards = prev_rewards_1 if agent_key == algaater_1.name else prev_rewards_2
            prev_opp_rewards = prev_rewards_2 if agent_key == algaater_1.name else prev_rewards_1
            agent_reward = reward_map[agent_key]
            proposed_total_payoff = agent_reward + proposed_payoff_to_go
            proportion_payoff = agent_reward / proposed_total_payoff if proposed_total_payoff != 0 else agent_reward / 0.000001
            opp_actions = opp_actions_1 if agent.player == P2 else opp_actions_2
            r1, r2 = (prev_reward_1, prev_reward_2) if agent_key == algaater_1.name else (prev_reward_2, prev_reward_1)
            new_assumptions = agent.update_expert(prev_rewards, prev_opp_rewards,
                                                                     round_num, agent_reward / (round_num + 1),
                                                                     proposed_total_payoff,
                                                                     agent_reward, n_remaining_rounds, state, r1, r2, ACTIONS, DESCRIPTION)

            if agent_key == algaater_1.name:
                # prev_short_term_1, prev_medium_term_1, prev_long_term_1 = short_term, medium_term, long_term
                prev_assumptions_1 = new_assumptions

            else:
                # prev_short_term_2, prev_medium_term_2, prev_long_term_2 = short_term, medium_term, long_term
                prev_assumptions_2 = new_assumptions

        if algaater_1.tracked_vector is not None:
            assert algaater_1.expert_to_use is not None
            with open(vector_file, 'a', newline='') as file:
                writer = csv.writer(file)
                row = np.concatenate([np.array([algaater_1.expert_to_use.name]), algaater_1.tracked_vector])
                writer.writerow(np.squeeze(row))

    total_rewards_1.append(epoch_rewards_1)
    total_rewards_2.append(epoch_rewards_2)

vals = []

for i in range(len(total_rewards_1)):
    final_reward_1, final_reward_2 = total_rewards_1[i][-1], total_rewards_2[i][-1]

    vals.extend([final_reward_1, final_reward_2])

compressed_rewards_df = pd.DataFrame(vals, columns=['Algaater'])
compressed_rewards_df.to_csv(f'../../analysis/{str(game)}/algaater_self_play{file_modifier}.csv')

test_results = np.array(total_rewards_1).reshape(n_epochs, -1)
opponent_test_results = np.array(total_rewards_2).reshape(n_epochs, -1)

mean_test_results = test_results.mean(axis=0)
mean_opponent_results = opponent_test_results.mean(axis=0)

x_vals = list(range(test_results.shape[1]))

plt.grid()
plt.plot(x_vals, mean_test_results, label='AlegAATr')
plt.plot(x_vals, mean_opponent_results, color='red', label='AlegAATr Mirror')
plt.title('Self Play Rewards')
plt.xlabel('Round #')
plt.ylabel('Rewards')
plt.legend(loc="upper left")
plt.savefig(f'../simulations/{str(game)}/algaater_self_play{file_modifier}.png', bbox_inches='tight')
plt.clf()

print(f'AlgAATer1: {mean_test_results[-1]}')
print(f'AlgAATer2: {mean_opponent_results[-1]}')
print(f'Combined: {(mean_test_results[-1] + mean_opponent_results[-1]) / 2}')
