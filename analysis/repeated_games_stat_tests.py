import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


# WARNING: BE SURE TO RUN THE CODE IN repeated_games_graphs.py (to generate/update the csv files required for these
# tests) BEFORE RUNNING THIS FILE

# NOTE: effect sizes are always calculated with AlegAATr as the first group and AlegAAATr as the second group

def _cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    return mean_diff / pooled_std


for game_name in ['prisoners_dilemma_game', 'chicken_game', 'coordination_game']:
    print('-----------------------------------------------------------------------------------------------------------')
    print(f'-------------------------------------- {game_name} --------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')

    # Train
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed.csv')
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    aleg_auto_rewards = df[df['Agent'] == 'AlegAAATr']['Rewards']
    t_stat, p_val = ttest_ind(aleg_rewards, aleg_auto_rewards)
    effect_size = _cohens_d(aleg_rewards, aleg_auto_rewards)

    print('TRAIN')
    print(f'p-value = {round(p_val, 3)}, effect size = {round(effect_size, 3)}\n')

    # Test
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_test.csv')
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    aleg_auto_rewards = df[df['Agent'] == 'AlegAAATr']['Rewards']
    t_stat, p_val = ttest_ind(aleg_rewards, aleg_auto_rewards)
    effect_size = _cohens_d(aleg_rewards, aleg_auto_rewards)

    print('TEST')
    print(f'p-value = {round(p_val, 3)}, effect size = {round(effect_size, 3)}\n')

    # Changers
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_test_changers.csv')
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    aleg_auto_rewards = df[df['Agent'] == 'AlegAAATr']['Rewards']
    t_stat, p_val = ttest_ind(aleg_rewards, aleg_auto_rewards)
    effect_size = _cohens_d(aleg_rewards, aleg_auto_rewards)

    print('CHANGERS')
    print(f'p-value = {round(p_val, 3)}, effect size = {round(effect_size, 3)}\n')

    # Self-play
    df = pd.read_csv(f'../analysis/{game_name}/combined_self_play.csv')
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    aleg_auto_rewards = df[df['Agent'] == 'AlegAAATr']['Rewards']
    t_stat, p_val = ttest_ind(aleg_rewards, aleg_auto_rewards)
    effect_size = _cohens_d(aleg_rewards, aleg_auto_rewards)

    print('SELF-PLAY')
    print(f'p-value = {round(p_val, 3)}, effect size = {round(effect_size, 3)}\n')

    # Intelligent
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_smart.csv')
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    aleg_auto_rewards = df[df['Agent'] == 'AlegAAATr']['Rewards']
    t_stat, p_val = ttest_ind(aleg_rewards, aleg_auto_rewards)
    effect_size = _cohens_d(aleg_rewards, aleg_auto_rewards)

    print('INTELLIGENT')
    print(f'p-value = {round(p_val, 3)}, effect size = {round(effect_size, 3)}')

    print('-----------------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
    print()
