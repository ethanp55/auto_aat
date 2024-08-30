import pandas as pd
import matplotlib.pyplot as plt


# Train (expert pool)
algaater_compressed = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed.csv')
algaater_compressed['Agent'] = ['AlegAATr'] * len(algaater_compressed)
algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed_auto.csv')
algaater_auto_compressed['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed)
algaater_auto_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

combined_df = pd.concat([algaater_compressed, algaater_auto_compressed], axis=0, ignore_index=True)
combined_df.to_csv('../analysis/prisoners_dilemma_game/combined_compressed.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.grid()
plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards']], labels=['AlgAATer', 'AlegAAATr'])
plt.xlabel('Agent', fontsize=16, fontweight='bold')
plt.ylabel('Rewards', fontsize=16, fontweight='bold')
plt.savefig(f'../analysis/prisoners_dilemma_game/agent_rewards.png', bbox_inches='tight')
plt.clf()

# Test (basic agents not in expert pool)
algaater_compressed_test = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed_test.csv')
algaater_compressed_test['Agent'] = ['AlegAATr'] * len(algaater_compressed_test)
algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed_test = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed_test_auto.csv')
algaater_auto_compressed_test['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed_test)
algaater_auto_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

combined_df_test = pd.concat([algaater_compressed_test, algaater_auto_compressed_test], axis=0, ignore_index=True)
combined_df_test.to_csv('../analysis/prisoners_dilemma_game/combined_compressed_test.csv')
combined_df_test.reset_index(drop=True, inplace=True)

plt.grid()
plt.boxplot([algaater_compressed_test['Rewards'], algaater_auto_compressed_test['Rewards']],
            labels=['AlgAATer', 'AlegAAATr'])
plt.xlabel('Agent', fontsize=16, fontweight='bold')
plt.ylabel('Rewards', fontsize=16, fontweight='bold')
plt.savefig(f'../analysis/prisoners_dilemma_game/agent_rewards_test.png', bbox_inches='tight')
plt.clf()

# Basic agents that change strategy periodically
algaater_compressed_test = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed_test_changers.csv')
algaater_compressed_test['Agent'] = ['AlegAATr'] * len(algaater_compressed_test)
algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed_test = \
    pd.read_csv('../analysis/prisoners_dilemma_game/algaater_compressed_test_changers_auto.csv')
algaater_auto_compressed_test['Agent'] = ['AlegAAATr'] * len(algaater_compressed_test)
algaater_auto_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

combined_df_test = pd.concat([algaater_compressed_test, algaater_auto_compressed_test], axis=0, ignore_index=True)
combined_df_test.to_csv('../analysis/prisoners_dilemma_game/combined_compressed_test_changers.csv')
combined_df_test.reset_index(drop=True, inplace=True)

plt.grid()
plt.boxplot([algaater_compressed_test['Rewards'], algaater_auto_compressed_test['Rewards']],
            labels=['AlegAATr', 'AlegAAATr'])
plt.xlabel('Agent', fontsize=16, fontweight='bold')
plt.ylabel('Rewards', fontsize=16, fontweight='bold')
plt.savefig(f'../analysis/prisoners_dilemma_game/agent_rewards_test_changers.png', bbox_inches='tight')
plt.clf()

# Self-play
algaater_compressed = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_self_play.csv')
algaater_compressed['Agent'] = ['AlegAATr'] * len(algaater_compressed)
algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_self_play_auto.csv')
algaater_auto_compressed['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed)
algaater_auto_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

combined_df = pd.concat([algaater_compressed, algaater_auto_compressed], axis=0, ignore_index=True)
combined_df.to_csv('../analysis/prisoners_dilemma_game/combined_self_play.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards']], labels=['AlegAATr', 'AlegAAATr'])
plt.xlabel('Agent', fontsize=16, fontweight='bold')
plt.ylabel('Rewards', fontsize=16, fontweight='bold')
plt.savefig(f'../analysis/prisoners_dilemma_game/agent_rewards_self_play.png', bbox_inches='tight')
plt.clf()

# Intelligent agents (BBL, EEE, S++)
algaater_compressed1 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_bbl_algaater.csv')
algaater_compressed1['Agent'] = ['AlegAATr'] * len(algaater_compressed1)
algaater_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_compressed2 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_eee_algaater.csv')
algaater_compressed2['Agent'] = ['AlegAATr'] * len(algaater_compressed2)
algaater_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_compressed3 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_spp_algaater.csv')
algaater_compressed3['Agent'] = ['AlegAATr'] * len(algaater_compressed3)
algaater_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed1 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_bbl_algaater_auto.csv')
algaater_auto_compressed1['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed1)
algaater_auto_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed2 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_eee_algaater_auto.csv')
algaater_auto_compressed2['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed2)
algaater_auto_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_auto_compressed3 = pd.read_csv('../analysis/prisoners_dilemma_game/algaater_vs_spp_algaater_auto.csv')
algaater_auto_compressed3['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed3)
algaater_auto_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_compressed = pd.concat([algaater_compressed1, algaater_compressed2, algaater_compressed3], axis=0,
                                ignore_index=True)

algaater_auto_compressed = pd.concat([algaater_auto_compressed1, algaater_auto_compressed2, algaater_auto_compressed3],
                                     axis=0, ignore_index=True)

combined_df = pd.concat([algaater_compressed, algaater_auto_compressed], axis=0, ignore_index=True)
combined_df.to_csv('../analysis/prisoners_dilemma_game/combined_compressed_smart.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.grid()
plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards']], labels=['AlegAATr', 'AlegAAATr'])
plt.xlabel('Agent', fontsize=16, fontweight='bold')
plt.ylabel('Rewards', fontsize=16, fontweight='bold')
plt.savefig(f'../analysis/prisoners_dilemma_game/agent_rewards_smart.png', bbox_inches='tight')
plt.clf()
