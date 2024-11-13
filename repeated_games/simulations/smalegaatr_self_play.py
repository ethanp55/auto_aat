from repeated_games.chicken_game import ChickenGame, baselines as chicken_baselines, ACTIONS as chicken_actions
from repeated_games.coordination_game import CoordinationGame, baselines as coord_baselines, ACTIONS as coord_actions
from repeated_games.prisoners_dilemma import PrisonersDilemma, baselines as pd_baselines, ACTIONS as pd_actions
from repeated_games.agents.smalegaatr import SMAlegAATr
from utils.utils import CHICKEN_E_DESCRIPTION, COORD_E_DESCRIPTION, P1, P2, PRISONERS_E_DESCRIPTION
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import csv

game = PrisonersDilemma()

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

vector_file = f'../../analysis/{str(game)}_vectors/SMalegAATr_self_play.csv'
with open(vector_file, 'w', newline='') as _:
    pass

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))

    epoch_rewards_1 = []
    epoch_rewards_2 = []

    smalgaatr_idx_1 = np.random.choice([P1, P2])
    smalgaatr_idx_2 = 1 - smalgaatr_idx_1

    smalegaatr_1 = SMAlegAATr('SMAlegAATr1', game, smalgaatr_idx_1)
    smalegaatr_2 = SMAlegAATr('SMAlegAATr2', game, smalgaatr_idx_2)

    n_rounds = min_rounds

    reward_map = {smalegaatr_1.name: 0, smalegaatr_2.name: 0}

    prev_reward_1 = 0
    prev_reward_2 = 0

    for round_num in range(n_rounds):
        game.reset()
        state = deepcopy(game.get_init_state())
        action_map = dict()
        opp_actions_1 = []
        opp_actions_2 = []

        key_agent_map = {smalegaatr_1.name: smalegaatr_1, smalegaatr_2.name: smalegaatr_2} if smalgaatr_idx_1 == P1 else \
            {smalegaatr_2.name: smalegaatr_2, smalegaatr_1.name: smalegaatr_1}

        rewards_1 = []
        rewards_2 = []

        while not state.is_terminal():
            for agent_key, agent in key_agent_map.items():
                agent_reward = prev_reward_1 if agent_key == smalegaatr_1.name else prev_reward_2
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                if agent_key == smalegaatr_1.name:
                    opp_action = agent_action1 if smalegaatr_1.player == P1 else agent_action2
                    opp_actions_2.append(opp_action)

                else:
                    opp_action = agent_action1 if smalegaatr_2.player == P1 else agent_action2
                    opp_actions_1.append(opp_action)

            updated_rewards_map, next_state = game.execute_agent_action(action_map)

            for agent_name, new_reward in updated_rewards_map.items():
                reward_map[agent_name] += new_reward

                if agent_name == smalegaatr_1.name:
                    rewards_1.append(new_reward)

                else:
                    rewards_2.append(new_reward)

            state = next_state

        prev_reward_1 = sum(rewards_1)
        prev_reward_2 = sum(rewards_2)

        epoch_rewards_1.append(reward_map[smalegaatr_1.name])
        epoch_rewards_2.append(reward_map[smalegaatr_2.name])

        smalegaatr_1.update_state(state, prev_reward_1, prev_reward_2)
        smalegaatr_2.update_state(state, prev_reward_1, prev_reward_2)

        if smalegaatr_1.tracked_vector is not None:
            assert smalegaatr_1.generator_in_use_name is not None
            with open(vector_file, 'a', newline='') as file:
                writer = csv.writer(file)
                row = np.concatenate([np.array([smalegaatr_1.generator_in_use_name]), smalegaatr_1.tracked_vector])
                writer.writerow(np.squeeze(row))

    total_rewards_1.append(epoch_rewards_1)
    total_rewards_2.append(epoch_rewards_2)

vals = []

for i in range(len(total_rewards_1)):
    final_reward_1, final_reward_2 = total_rewards_1[i][-1], total_rewards_2[i][-1]

    vals.extend([final_reward_1, final_reward_2])

compressed_rewards_df = pd.DataFrame(vals, columns=['SMAlegAATr'])
compressed_rewards_df.to_csv(f'../../analysis/{str(game)}/smalgaatr_self_play.csv')

test_results = np.array(total_rewards_1).reshape(n_epochs, -1)
opponent_test_results = np.array(total_rewards_2).reshape(n_epochs, -1)

mean_test_results = test_results.mean(axis=0)
mean_opponent_results = opponent_test_results.mean(axis=0)

x_vals = list(range(test_results.shape[1]))

plt.grid()
plt.plot(x_vals, mean_test_results, label='SMAlegAATr')
plt.plot(x_vals, mean_opponent_results, color='red', label='SMAlegAATr Mirror')
plt.title('Self Play Rewards')
plt.xlabel('Round #')
plt.ylabel('Rewards')
plt.legend(loc="upper left")
plt.savefig(f'../simulations/{str(game)}/smalgaatr_self_play.png', bbox_inches='tight')
plt.clf()

print(f'SMAlegAATr1: {mean_test_results[-1]}')
print(f'SMAlegAATr2: {mean_opponent_results[-1]}')
print(f'Combined: {(mean_test_results[-1] + mean_opponent_results[-1]) / 2}')
