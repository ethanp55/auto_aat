import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# WARNING: BE SURE TO RUN THE CODE IN repeated_games_graphs.py (to generate/update the csv files required for these
# tests) BEFORE RUNNING THIS FILE

# NOTE: effect sizes are always calculated with AlegAATr as the first group and AlegAAATr as the second group

def _cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    return mean_diff / pooled_std


# for game_name in ['prisoners_dilemma_game', 'chicken_game', 'coordination_game']:
for game_name in ['chicken_game']:
    print('-----------------------------------------------------------------------------------------------------------')
    print(f'-------------------------------------- {game_name} --------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')

    all_vals_df = pd.DataFrame([])

    # Train
    print('TRAIN')
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed.csv')
    all_vals_df = pd.concat([all_vals_df, df], axis=0, ignore_index=True)
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    for name in df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = df[df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=df['Rewards'], groups=df['Agent'], alpha=0.05))

    # Test
    print('\nTEST')
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_test.csv')
    all_vals_df = pd.concat([all_vals_df, df], axis=0, ignore_index=True)
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    for name in df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = df[df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=df['Rewards'], groups=df['Agent'], alpha=0.05))

    # Changers
    print('\nCHANGERS')
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_test_changers.csv')
    all_vals_df = pd.concat([all_vals_df, df], axis=0, ignore_index=True)
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    for name in df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = df[df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=df['Rewards'], groups=df['Agent'], alpha=0.05))

    # Self-play
    print('\nSELF-PLAY')
    df = pd.read_csv(f'../analysis/{game_name}/combined_self_play.csv')
    all_vals_df = pd.concat([all_vals_df, df], axis=0, ignore_index=True)
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    for name in df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = df[df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=df['Rewards'], groups=df['Agent'], alpha=0.05))

    # Intelligent
    print('\nINTELLIGENT')
    df = pd.read_csv(f'../analysis/{game_name}/combined_compressed_smart.csv')
    all_vals_df = pd.concat([all_vals_df, df], axis=0, ignore_index=True)
    aleg_rewards = df[df['Agent'] == 'AlegAATr']['Rewards']
    for name in df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = df[df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=df['Rewards'], groups=df['Agent'], alpha=0.05))

    # Overall
    print('\nOVERALL')
    aleg_rewards = all_vals_df[all_vals_df['Agent'] == 'AlegAATr']['Rewards']
    for name in all_vals_df['Agent'].unique():
        if name == 'AlegAATr':
            continue
        other_rewards = all_vals_df[all_vals_df['Agent'] == name]['Rewards']
        d = _cohens_d(aleg_rewards, other_rewards)
        print(f'AlegAATr vs. {name}: {round(d, 3)}')
    print(pairwise_tukeyhsd(endog=all_vals_df['Rewards'], groups=all_vals_df['Agent'], alpha=0.05))

    print('-----------------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
    print()
