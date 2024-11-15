import repeated_games.agents.chicken_game_specific_agents as chicken_game_specific_agents
import repeated_games.agents.coordination_game_specific_agents as coordination_game_specific_agents
import repeated_games.agents.prisoners_dilemma_specific_agents as prisoners_dilemma_specific_agents
from repeated_games.agents.smalegaatr import SMAlegAATr
from repeated_games.chicken_game import ChickenGame, baselines as chicken_baselines, ACTIONS as chicken_actions
from repeated_games.coordination_game import CoordinationGame, baselines as coord_baselines, ACTIONS as coord_actions
from repeated_games.prisoners_dilemma import PrisonersDilemma, baselines as pd_baselines, ACTIONS as pd_actions
from repeated_games.agents.alegaatr import AlegAATr, ESTIMATES_LOOKBACK, Assumptions
from utils.utils import CHICKEN_E_DESCRIPTION, COORD_E_DESCRIPTION, P1, P2, PRISONERS_E_DESCRIPTION
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from repeated_games.agents.spp import SPP
from repeated_games.agents.eee import EEE
from repeated_games.agents.folk_egal import FolkEgalAgent, FolkEgalPunishAgent
import csv

game = ChickenGame()

baselines = pd_baselines if str(game) == 'prisoners_dilemma_game' else \
    (chicken_baselines if str(game) == 'chicken_game' else coord_baselines)
ACTIONS = pd_actions if str(game) == 'prisoners_dilemma_game' else \
    (chicken_actions if str(game) == 'chicken_game' else coord_actions)
DESCRIPTION = PRISONERS_E_DESCRIPTION if str(game) == 'prisoners_dilemma_game' else \
    (CHICKEN_E_DESCRIPTION if str(game) == 'chicken_game' else COORD_E_DESCRIPTION)

EXPERT_SET_NAMES = ['CoopOpp', 'CoopPunishOpp', 'BullyOpp', 'BullyPunishOpp', 'BulliedOpp', 'MinimaxOpp', 'CfrOpp']
OTHER_NAMES = ['Random', 'GreedyNeg', 'CoopGreedy', 'RoundNum', 'RoundNum2', 'RoundNum3']
CHANGER_NAMES = ['RoundNum', 'RoundNum2', 'RoundNum3']


# Create opponent agents
def create_opponent_agents(player_idx):
    initial_state = game.get_init_state()
    game_name = str(game)

    # Experts in AlgAATer's pool
    coop_agent = FolkEgalAgent('CoopOpp', 1, 1, initial_state, game_name, read_from_file=True, player=player_idx)
    coop_punish_agent = FolkEgalPunishAgent('CoopPunishOpp', coop_agent, game_name, game)
    bully_agent = FolkEgalAgent('BullyOpp', 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                                specific_policy=True, p1_weight=1.0, player=player_idx)
    bully_punish_agent = FolkEgalPunishAgent('BullyPunishOpp', bully_agent, game_name, game)

    # Other agents
    eee_experts = AlegAATr.create_aat_experts(game, player_idx)
    spp = SPP('RoundNum3S++', game, player_idx)
    eee = EEE('RoundNum3EEE', eee_experts, player_idx, demo=True)

    if game_name == 'prisoners_dilemma_game':
        random_agent = prisoners_dilemma_specific_agents.Random('Random', player_idx)
        greedy_neg_agent = prisoners_dilemma_specific_agents.GreedyUntilNegative('GreedyNeg', player_idx)
        coop_greedy_agent = prisoners_dilemma_specific_agents.CoopOrGreedy('CoopGreedy', player_idx)
        round_num_agent = prisoners_dilemma_specific_agents.RoundNum('RoundNum', player_idx)
        round_num2_agent = prisoners_dilemma_specific_agents.RoundNum2('RoundNum2', player_idx)
        round_num3_agent = prisoners_dilemma_specific_agents.RoundNum3('RoundNum3', player_idx, spp, eee)

    elif game_name == 'chicken_game':
        random_agent = chicken_game_specific_agents.Random('Random', player_idx)
        greedy_neg_agent = chicken_game_specific_agents.GreedyUntilNegative('GreedyNeg', player_idx)
        coop_greedy_agent = chicken_game_specific_agents.CoopOrGreedy('CoopGreedy', player_idx)
        round_num_agent = chicken_game_specific_agents.RoundNum('RoundNum', player_idx)
        round_num2_agent = chicken_game_specific_agents.RoundNum2('RoundNum2', player_idx)
        round_num3_agent = chicken_game_specific_agents.RoundNum3('RoundNum3', player_idx, spp, eee)

    else:
        random_agent = coordination_game_specific_agents.Random('Random', player_idx)
        greedy_neg_agent = coordination_game_specific_agents.GreedyUntilNegative('GreedyNeg', player_idx)
        coop_greedy_agent = coordination_game_specific_agents.CoopOrGreedy('CoopGreedy', player_idx)
        round_num_agent = coordination_game_specific_agents.RoundNum('RoundNum', player_idx)
        round_num2_agent = coordination_game_specific_agents.RoundNum2('RoundNum2', player_idx)
        round_num3_agent = coordination_game_specific_agents.RoundNum3('RoundNum3', player_idx, spp, eee)

    opponents = {coop_punish_agent.name: coop_punish_agent, bully_agent.name: bully_agent, coop_greedy_agent.name: coop_greedy_agent,
                 random_agent.name: random_agent, bully_punish_agent.name: bully_punish_agent, greedy_neg_agent.name:
                     greedy_neg_agent, round_num_agent.name: round_num_agent,
                 round_num2_agent.name: round_num2_agent, round_num3_agent.name: round_num3_agent}

    # opponents = {coop_punish.name: coop_punish}

    return opponents


create_graphs = True
save_data = True

n_epochs = 50
min_rounds = 50
max_rounds = 100
possible_rounds = list(range(min_rounds, max_rounds + 1))
total_rewards = {}
total_opp_rewards = {}
vector_file = f'../../analysis/{str(game)}_vectors/SMAlegAATr_basic.csv'
with open(vector_file, 'w', newline='') as _:
    pass

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))
    smalegaatr_idx = np.random.choice([P1, P2])
    opponent_idx = 1 - smalegaatr_idx

    opponents = create_opponent_agents(opponent_idx)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    for opponent_key in opponents.keys():
        smalegaatr = SMAlegAATr('SMAlegAATr', game, smalegaatr_idx)

        opponent_agent = deepcopy(opponents[opponent_key])
        algaater_rewards = []
        opp_rewards = []
        reward_map = {opponent_key: 0, smalegaatr.name: 0}

        prev_reward_1 = 0
        prev_reward_2 = 0

        for round_num in range(n_rounds):
            game.reset()
            state = deepcopy(game.get_init_state())
            action_map = dict()
            opp_actions = []
            actions = []

            key_agent_map = {smalegaatr.name: smalegaatr, opponent_key: opponent_agent} if smalegaatr_idx == P1 else \
                {opponent_key: opponent_agent, smalegaatr.name: smalegaatr}

            rewards_1 = []
            rewards_2 = []

            while not state.is_terminal():
                for agent_key, agent in key_agent_map.items():
                    agent_reward = prev_reward_1 if agent_key == smalegaatr.name else prev_reward_2
                    agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                    action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                    if agent_key == opponent_key:
                        opp_action = agent_action1 if opponent_agent.player == P1 else agent_action2
                        opp_actions.append(opp_action)

                    else:
                        our_action = agent_action1 if smalegaatr.player == P1 else agent_action2
                        actions.append(our_action)

                    # elif agent_key != opponent_key:
                    #     algaater.assumption_checker.act(state, agent_reward, round_num)

                updated_rewards_map, next_state = game.execute_agent_action(action_map)

                for agent_name, new_reward in updated_rewards_map.items():
                    reward_map[agent_name] += new_reward

                    if agent_name == smalegaatr.name:
                        rewards_1.append(new_reward)

                    else:
                        rewards_2.append(new_reward)

                prev_state = deepcopy(state)
                state = next_state

            prev_reward_1 = sum(rewards_1)
            prev_reward_2 = sum(rewards_2)
            smalegaatr.update_state(state, prev_reward_1, prev_reward_2)
            agent_reward = reward_map[smalegaatr.name]
            algaater_rewards.append(agent_reward)
            opp_rewards.append(reward_map[opponent_key])

            if smalegaatr.tracked_vector is not None:
                assert smalegaatr.generator_in_use_name is not None
                with open(vector_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = np.concatenate([np.array([smalegaatr.generator_in_use_name]), smalegaatr.tracked_vector])
                    writer.writerow(np.squeeze(row))

        total_rew = total_rewards.get(opponent_key, [])
        total_rew.append(algaater_rewards)
        total_rewards[opponent_key] = total_rew
        total_opp_rew = total_opp_rewards.get(opponent_key, [])
        total_opp_rew.append(opp_rewards)
        total_opp_rewards[opponent_key] = total_opp_rew

if save_data:
    vals = []
    vals_test = []
    vals_test_changers = []

    for expert_key, rewards in total_rewards.items():
        for epoch_rewards in rewards:
            if expert_key in EXPERT_SET_NAMES:
                vals.append(epoch_rewards[-1])

            else:
                vals_test.append(epoch_rewards[-1])

            if expert_key in CHANGER_NAMES:
                vals_test_changers.append(epoch_rewards[-1])

    compressed_rewards_df = pd.DataFrame(vals, columns=['SMAlegAATr'])
    compressed_rewards_test_df = pd.DataFrame(vals_test, columns=['SMAlegAATr'])
    compressed_rewards_test_changers_df = pd.DataFrame(vals_test_changers, columns=['SMAlegAATr'])

    columns = []
    columns_test = []
    columns_test_changers = []
    vals = []
    vals_test = []
    vals_test_changers = []

    for agent_name, rewards in total_rewards.items():
        agent_epoch_rewards = []

        for epoch_rewards in rewards:
            agent_epoch_rewards.append(epoch_rewards[-1])

        if agent_name in EXPERT_SET_NAMES:
            vals.append(agent_epoch_rewards)
            columns.append(agent_name)

        else:
            vals_test.append(agent_epoch_rewards)
            columns_test.append(agent_name)

        if agent_name in CHANGER_NAMES:
            vals_test_changers.append(agent_epoch_rewards)
            columns_test_changers.append(agent_name)

    full_rewards_df = pd.DataFrame(zip(*vals), columns=columns)
    full_rewards_test_df = pd.DataFrame(zip(*vals_test), columns=columns_test)
    full_rewards_test_changers_df = pd.DataFrame(zip(*vals_test_changers), columns=columns_test_changers)

    compressed_rewards_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_compressed.csv')
    compressed_rewards_test_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_compressed_test.csv')
    compressed_rewards_test_changers_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_compressed_test_changers.csv')
    full_rewards_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_full.csv')
    full_rewards_test_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_full_test.csv')
    full_rewards_test_changers_df.to_csv(f'../../analysis/{str(game)}/smalegaatr_full_test_changers.csv')

if create_graphs:
    for opponent_key in opponents.keys():
        algaater_epoch_rewards = np.array(total_rewards[opponent_key]).reshape(n_epochs, -1)
        opp_epoch_rewards = np.array(total_opp_rewards[opponent_key]).reshape(n_epochs, -1)

        algaater_mean_rewards = algaater_epoch_rewards.mean(axis=0)
        opp_mean_rewards = opp_epoch_rewards.mean(axis=0)

        x_vals = list(range(algaater_epoch_rewards.shape[1]))

        plt.grid()
        plt.plot(x_vals, algaater_mean_rewards, label='SMAlegAATr')
        plt.plot(x_vals, opp_mean_rewards, color='red', label=opponent_key)
        plt.title('SMAlegAATr vs. ' + str(opponent_key))
        plt.xlabel('Round #')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
        plt.savefig(f'../simulations/{str(game)}/smalegaatr_vs_{opponent_key}.png', bbox_inches='tight')
        plt.clf()
