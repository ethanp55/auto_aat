import pandas as pd
import matplotlib.pyplot as plt


# for game_name in ['prisoners_dilemma_game', 'chicken_game', 'coordination_game']:
for game_name in ['coordination_game']:
    all_vals_df = pd.DataFrame([])

    # Train (expert pool)
    algaater_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_compressed.csv')
    algaater_compressed['Agent'] = ['AlegAATr'] * len(algaater_compressed)
    algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_auto.csv')
    algaater_auto_compressed['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed)
    algaater_auto_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_auto_tuned.csv')
    algaater_auto_tuned_compressed['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed)
    algaater_auto_tuned_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    smalegaatr_compressed = pd.read_csv(f'../analysis/{game_name}/smalegaatr_compressed.csv')
    smalegaatr_compressed['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed)
    smalegaatr_compressed.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    combined_df = pd.concat([algaater_compressed, algaater_auto_compressed, algaater_auto_tuned_compressed,
                             smalegaatr_compressed], axis=0, ignore_index=True)
    combined_df.to_csv(f'../analysis/{game_name}/combined_compressed.csv')
    combined_df.reset_index(drop=True, inplace=True)
    all_vals_df = pd.concat([all_vals_df, combined_df], axis=0, ignore_index=True)

    plt.grid()
    plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards'],
                 algaater_auto_tuned_compressed['Rewards'], smalegaatr_compressed['Rewards']],
                labels=['AlgAATer', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards.png', bbox_inches='tight')
    plt.clf()

    # Test (basic agents not in expert pool)
    algaater_compressed_test = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test.csv')
    algaater_compressed_test['Agent'] = ['AlegAATr'] * len(algaater_compressed_test)
    algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed_test = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test_auto.csv')
    algaater_auto_compressed_test['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed_test)
    algaater_auto_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed_test = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test_auto_tuned.csv')
    algaater_auto_tuned_compressed_test['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed_test)
    algaater_auto_tuned_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    smalegaatr_compressed_test = pd.read_csv(f'../analysis/{game_name}/smalegaatr_compressed_test.csv')
    smalegaatr_compressed_test['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed_test)
    smalegaatr_compressed_test.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    combined_df_test = pd.concat([algaater_compressed_test, algaater_auto_compressed_test,
                                  algaater_auto_tuned_compressed_test, smalegaatr_compressed_test], axis=0,
                                 ignore_index=True)
    combined_df_test.to_csv(f'../analysis/{game_name}/combined_compressed_test.csv')
    combined_df_test.reset_index(drop=True, inplace=True)
    all_vals_df = pd.concat([all_vals_df, combined_df_test], axis=0, ignore_index=True)

    plt.grid()
    plt.boxplot([algaater_compressed_test['Rewards'], algaater_auto_compressed_test['Rewards'],
                 algaater_auto_tuned_compressed_test['Rewards'], smalegaatr_compressed_test['Rewards']],
                labels=['AlgAATer', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_test.png', bbox_inches='tight')
    plt.clf()

    # Basic agents that change strategy periodically
    algaater_compressed_test = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test_changers.csv')
    algaater_compressed_test['Agent'] = ['AlegAATr'] * len(algaater_compressed_test)
    algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed_test = \
        pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test_changers_auto.csv')
    algaater_auto_compressed_test['Agent'] = ['AlegAAATr'] * len(algaater_compressed_test)
    algaater_auto_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed_test = pd.read_csv(f'../analysis/{game_name}/algaater_compressed_test_changers_auto_tuned.csv')
    algaater_auto_tuned_compressed_test['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed_test)
    algaater_auto_tuned_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    smalegaatr_compressed_test = pd.read_csv(f'../analysis/{game_name}/smalegaatr_compressed_test_changers.csv')
    smalegaatr_compressed_test['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed_test)
    smalegaatr_compressed_test.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    combined_df_test = pd.concat([algaater_compressed_test, algaater_auto_compressed_test,
                                  algaater_auto_tuned_compressed_test, smalegaatr_compressed_test], axis=0,
                                 ignore_index=True)
    combined_df_test.to_csv(f'../analysis/{game_name}/combined_compressed_test_changers.csv')
    combined_df_test.reset_index(drop=True, inplace=True)
    all_vals_df = pd.concat([all_vals_df, combined_df_test], axis=0, ignore_index=True)

    plt.grid()
    plt.boxplot([algaater_compressed_test['Rewards'], algaater_auto_compressed_test['Rewards'],
                 algaater_auto_tuned_compressed_test['Rewards'], smalegaatr_compressed_test['Rewards']],
                labels=['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_test_changers.png', bbox_inches='tight')
    plt.clf()

    # Self-play
    algaater_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_self_play.csv')
    algaater_compressed['Agent'] = ['AlegAATr'] * len(algaater_compressed)
    algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_self_play_auto.csv')
    algaater_auto_compressed['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed)
    algaater_auto_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed = pd.read_csv(f'../analysis/{game_name}/algaater_self_play_auto_tuned.csv')
    algaater_auto_tuned_compressed['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed)
    algaater_auto_tuned_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    smalegaatr_compressed_test = pd.read_csv(f'../analysis/{game_name}/smalgaatr_self_play.csv')
    smalegaatr_compressed_test['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed_test)
    smalegaatr_compressed_test.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    combined_df = pd.concat([algaater_compressed, algaater_auto_compressed, algaater_auto_tuned_compressed,
                             smalegaatr_compressed_test], axis=0, ignore_index=True)
    combined_df.to_csv(f'../analysis/{game_name}/combined_self_play.csv')
    combined_df.reset_index(drop=True, inplace=True)
    all_vals_df = pd.concat([all_vals_df, combined_df], axis=0, ignore_index=True)

    plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards'],
                 algaater_auto_tuned_compressed['Rewards'], smalegaatr_compressed_test['Rewards']],
                labels=['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_self_play.png', bbox_inches='tight')
    plt.clf()

    # Intelligent agents (BBL, EEE, S++)
    algaater_compressed1 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_bbl_algaater.csv')
    algaater_compressed1['Agent'] = ['AlegAATr'] * len(algaater_compressed1)
    algaater_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_compressed2 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_eee_algaater.csv')
    algaater_compressed2['Agent'] = ['AlegAATr'] * len(algaater_compressed2)
    algaater_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_compressed3 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_spp_algaater.csv')
    algaater_compressed3['Agent'] = ['AlegAATr'] * len(algaater_compressed3)
    algaater_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed1 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_bbl_algaater_auto.csv')
    algaater_auto_compressed1['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed1)
    algaater_auto_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed2 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_eee_algaater_auto.csv')
    algaater_auto_compressed2['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed2)
    algaater_auto_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_compressed3 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_spp_algaater_auto.csv')
    algaater_auto_compressed3['Agent'] = ['AlegAAATr'] * len(algaater_auto_compressed3)
    algaater_auto_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed1 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_bbl_algaater_auto_tuned.csv')
    algaater_auto_tuned_compressed1['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed1)
    algaater_auto_tuned_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed2 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_eee_algaater_auto_tuned.csv')
    algaater_auto_tuned_compressed2['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed2)
    algaater_auto_tuned_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    algaater_auto_tuned_compressed3 = pd.read_csv(f'../analysis/{game_name}/algaater_vs_spp_algaater_auto_tuned.csv')
    algaater_auto_tuned_compressed3['Agent'] = ['AlegAAATTr'] * len(algaater_auto_tuned_compressed3)
    algaater_auto_tuned_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

    smalegaatr_compressed1 = pd.read_csv(f'../analysis/{game_name}/smalegaatr_vs_bbl_smalegaatr.csv')
    smalegaatr_compressed1['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed1)
    smalegaatr_compressed1.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    smalegaatr_compressed2 = pd.read_csv(f'../analysis/{game_name}/smalegaatr_vs_eee_smalegaatr.csv')
    smalegaatr_compressed2['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed2)
    smalegaatr_compressed2.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    smalegaatr_compressed3 = pd.read_csv(f'../analysis/{game_name}/smalegaatr_vs_spp_smalegaatr.csv')
    smalegaatr_compressed3['Agent'] = ['SMAlegAATr'] * len(smalegaatr_compressed3)
    smalegaatr_compressed3.rename(columns={'SMAlegAATr': 'Rewards'}, inplace=True)

    algaater_compressed = pd.concat([algaater_compressed1, algaater_compressed2, algaater_compressed3], axis=0,
                                    ignore_index=True)

    algaater_auto_compressed = pd.concat([algaater_auto_compressed1, algaater_auto_compressed2,
                                          algaater_auto_compressed3], axis=0, ignore_index=True)

    algaater_auto_tuned_compressed = pd.concat([algaater_auto_tuned_compressed1, algaater_auto_tuned_compressed2,
                                                algaater_auto_tuned_compressed3], axis=0, ignore_index=True)

    smalegaatr_compressed = pd.concat([smalegaatr_compressed1, smalegaatr_compressed2, smalegaatr_compressed3],
                                      axis=0, ignore_index=True)

    combined_df = pd.concat([algaater_compressed, algaater_auto_compressed, algaater_auto_tuned_compressed,
                             smalegaatr_compressed], axis=0, ignore_index=True)
    combined_df.to_csv(f'../analysis/{game_name}/combined_compressed_smart.csv')
    combined_df.reset_index(drop=True, inplace=True)
    all_vals_df = pd.concat([all_vals_df, combined_df], axis=0, ignore_index=True)

    plt.grid()
    plt.boxplot([algaater_compressed['Rewards'], algaater_auto_compressed['Rewards'],
                 algaater_auto_tuned_compressed['Rewards'], smalegaatr_compressed['Rewards']],
                labels=['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_smart.png', bbox_inches='tight')
    plt.clf()

    # Overall
    plt.grid()
    plt.boxplot([all_vals_df[all_vals_df['Agent'] == 'AlegAATr']['Rewards'],
                 all_vals_df[all_vals_df['Agent'] == 'AlegAAATr']['Rewards'],
                 all_vals_df[all_vals_df['Agent'] == 'AlegAAATTr']['Rewards'],
                 all_vals_df[all_vals_df['Agent'] == 'SMAlegAATr']['Rewards']],
                labels=['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr'])
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Rewards', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_overall.png', bbox_inches='tight')
    plt.clf()

    average_rewards_by_alg = all_vals_df.groupby('Agent')['Rewards'].agg(
        ['mean', 'sem']).reset_index()
    plt.figure(figsize=(10, 3))
    plt.grid()
    bar_colors = ['blue', 'green', 'red', 'orange']
    plt.bar(average_rewards_by_alg['Agent'], average_rewards_by_alg['mean'],
            yerr=average_rewards_by_alg['sem'], capsize=5, color=bar_colors)
    plt.xlabel('Agent', fontsize=16, fontweight='bold')
    plt.ylabel('Reward', fontsize=16, fontweight='bold')
    plt.savefig(f'../analysis/{game_name}/agent_rewards_overall_bars.png', bbox_inches='tight')
    plt.clf()
