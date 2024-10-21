from repeated_games.chicken_game import ChickenGame, baselines as chicken_baselines, ACTIONS as chicken_actions
from repeated_games.coordination_game import CoordinationGame, baselines as coord_baselines, ACTIONS as coord_actions
from repeated_games.prisoners_dilemma import PrisonersDilemma, baselines as pd_baselines, ACTIONS as pd_actions
from repeated_games.agents.alegaatr import AlegAATr, ESTIMATES_LOOKBACK, Assumptions
from repeated_games.agents.eee import EEE
from utils.utils import CHICKEN_E_DESCRIPTION, COORD_E_DESCRIPTION, P1, P2, PRISONERS_E_DESCRIPTION
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

game = ChickenGame()

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

use_auto_aat = True

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))

    epoch_rewards_1 = []
    epoch_rewards_2 = []

    algaater_idx = np.random.choice([P1, P2])
    eee_idx = 1 - algaater_idx
    eee_experts = AlegAATr.create_aat_experts(game, eee_idx)

    algaater = AlegAATr('Algaater', game, algaater_idx, baselines, use_auto_aat=use_auto_aat)
    eee = EEE('EEE', eee_experts, eee_idx, demo=True)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    reward_map = {algaater.name: 0, eee.name: 0}
    prev_rewards_1 = deque(maxlen=ESTIMATES_LOOKBACK)
    prev_rewards_2 = deque(maxlen=ESTIMATES_LOOKBACK)

    # prev_short_term = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_medium_term = Assumptions(0, 0, 0, 0, 0, 0, 0)
    # prev_long_term = Assumptions(0, 0, 0, 0, 0, 0, 0)
    prev_assumptions = Assumptions(0, 0, 0, 0, 0, 0, 0)

    prev_reward_1 = 0
    prev_reward_2 = 0

    for round_num in range(n_rounds):
        game.reset()
        state = deepcopy(game.get_init_state())
        action_map = dict()
        opp_actions_1 = []
        opp_actions_2 = []

        key_agent_map = {algaater.name: algaater, eee.name: eee} if algaater_idx == P1 else \
            {eee.name: eee, algaater.name: algaater}

        rewards_1 = []
        rewards_2 = []

        while not state.is_terminal():
            for agent_key, agent in key_agent_map.items():
                agent_reward = prev_reward_1 if agent_key == algaater.name else prev_reward_2
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                if agent_key == algaater.name:
                    # algaater_1.assumption_checker.act(state, agent_reward, round_num)

                    if state.turn is None or state.turn == algaater_idx:
                        opp_action = agent_action1 if algaater.player == P1 else agent_action2
                        opp_actions_2.append(opp_action)

                else:
                    # algaater_2.assumption_checker.act(state, agent_reward, round_num)

                    if state.turn is None or state.turn == eee_idx:
                        opp_action = agent_action1 if eee.player == P1 else agent_action2
                        opp_actions_1.append(opp_action)

            updated_rewards_map, next_state = game.execute_agent_action(action_map)

            for agent_name, new_reward in updated_rewards_map.items():
                reward_map[agent_name] += new_reward

                if agent_name == algaater.name:
                    rewards_1.append(new_reward)

                else:
                    rewards_2.append(new_reward)

            state = next_state

        prev_reward_1 = sum(rewards_1)
        prev_reward_2 = sum(rewards_2)
        prev_rewards_1.append(prev_reward_1)
        epoch_rewards_1.append(reward_map[algaater.name])
        prev_rewards_2.append(prev_reward_2)
        epoch_rewards_2.append(reward_map[eee.name])
        proposed_avg_payoff = baselines[algaater.expert_to_use.name]
        n_remaining_rounds = n_rounds - round_num - 1
        proposed_payoff_to_go = proposed_avg_payoff * n_remaining_rounds

        agent_reward = reward_map[algaater.name]
        proposed_total_payoff = agent_reward + proposed_payoff_to_go
        proportion_payoff = agent_reward / proposed_total_payoff if proposed_total_payoff != 0 else agent_reward / 0.000001
        # short_term, medium_term, long_term = algaater.update_expert(prev_short_term, prev_medium_term, prev_long_term,
        #                                                             prev_rewards_1, prev_rewards_2,
        #                                                             round_num, proportion_payoff,
        #                                                             proposed_total_payoff,
        #                                                             agent_reward, n_remaining_rounds)
        new_assumptions = algaater.update_expert(prev_rewards_1, prev_rewards_2, round_num,
                                                 (agent_reward / (round_num + 1)),
                                                 proposed_total_payoff, agent_reward, n_remaining_rounds, state, prev_reward_1, prev_reward_2, ACTIONS, DESCRIPTION)

        # prev_short_term, prev_medium_term, prev_long_term = short_term, medium_term, long_term

        prev_assumptions = deepcopy(new_assumptions)

    total_rewards_1.append(epoch_rewards_1)
    total_rewards_2.append(epoch_rewards_2)

vals_1, vals_2 = [], []

for i in range(len(total_rewards_1)):
    final_reward_1, final_reward_2 = total_rewards_1[i][-1], total_rewards_2[i][-1]

    vals_1.append(final_reward_1)
    vals_2.append(final_reward_2)

file_modifier = '_auto' if use_auto_aat else ''

compressed_rewards_df = pd.DataFrame(vals_1, columns=['Algaater'])
compressed_rewards_df.to_csv(f'../../analysis/{str(game)}/algaater_vs_eee_algaater{file_modifier}.csv')

compressed_rewards_df_opp = pd.DataFrame(vals_2, columns=['EEE'])
compressed_rewards_df_opp.to_csv(f'../../analysis/{str(game)}/algaater_vs_eee_eee{file_modifier}.csv')

test_results = np.array(total_rewards_1).reshape(n_epochs, -1)
opponent_test_results = np.array(total_rewards_2).reshape(n_epochs, -1)

mean_test_results = test_results.mean(axis=0)
mean_opponent_results = opponent_test_results.mean(axis=0)

x_vals = list(range(test_results.shape[1]))

plt.grid()
plt.plot(x_vals, mean_test_results, label='AlegAATr')
plt.plot(x_vals, mean_opponent_results, color='red', label='EEE')
plt.title('AlegAATr vs. EEE')
plt.xlabel('Round #')
plt.ylabel('Rewards ($)')
plt.legend(loc="upper left")
plt.savefig(f'../simulations/{str(game)}/algaater_vs_eee{file_modifier}.png', bbox_inches='tight')
plt.clf()

print(f'AlgAATer: {mean_test_results[-1]}')
print(f'EEE: {mean_opponent_results[-1]}')
