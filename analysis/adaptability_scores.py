from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


NAMES = ['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAAATr']
COLORS = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4']


# Forex
def forex_scores() -> None:
    all_profits = {}
    directory = '../analysis/forex_results/final_balances_csv/'
    file_list = os.listdir(directory)

    pair_time_frame_combos = [f'{currency_pair}_{time_frame}' for currency_pair in CURRENCY_PAIRS for time_frame in
                              TIME_FRAMES]
    results_by_pair_time_frame = {}

    for pair_time_frame in pair_time_frame_combos:
        results_by_pair_time_frame[pair_time_frame] = {}

    # Extract the data
    for file_name in file_list:
        df = pd.read_csv(f'{directory}{file_name}')

        for strategy in df.columns[1:]:
            profit = df.loc[df.index[0], strategy] - 10000
            name = 'SMAlegAAATr' if strategy == 'SMAlegAATr' else strategy
            all_profits[name] = all_profits.get(name, []) + [profit]

    minimax_val, lowest_reward = 0, -10000
    defect_scores = {}
    for strategy in NAMES:
        for profit in all_profits[strategy]:
            regret = (0 - lowest_reward) - (profit - lowest_reward)
            val = 1 - max((regret / (0 - lowest_reward)), 0)
            assert 0 <= val <= 1
            defect_scores[strategy] = defect_scores.get(strategy, []) + [val]

    print('FOREX:')
    avgs, ses = [], []
    for strategy, scores in defect_scores.items():
        avg = np.array(scores).mean()
        avgs.append(avg)
        ses.append(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        print(f'{strategy}: d = {avg}')

    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.bar(NAMES, avgs, yerr=ses, capsize=5, color=COLORS)
    plt.xlabel('Algorithm', fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=18, fontweight='bold')
    plt.savefig(f'../analysis/adaptability_plots/forex_d.png', bbox_inches='tight')
    plt.clf()


def pursuit_scores() -> None:
    # Get the data
    folder = '../analysis/stag_hare_results/'
    names, grid_dims, opponent_types = [], [], []
    final_rewards, final_reward_sums = [], []

    for file in os.listdir(folder):
        agent_name = file.split('_')[0]
        height = file.split('h=')[1].split('_')[0]
        width = file.split('w=')[1].split('.')[0]
        opp_type = file[len(agent_name) + 1:].split('_')[0]
        data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)
        if data.shape[0] == 0:
            continue
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        for row in data:
            # Add the condition info
            names.append(agent_name)
            grid_dims.append(f'{height}_{width}')
            opponent_types.append(opp_type)

            # Calculate and add the results
            agent_reward = row[-1]
            final_rewards.append(agent_reward)
            final_reward_sums.append(sum(row))

    df = pd.DataFrame(
        {
            'algorithm': names,
            'grid_dims': grid_dims,
            'opponent_type': opponent_types,
            'reward_sum': final_reward_sums,
            'agent_final_reward': final_rewards,
        }
    )
    df['algorithm'] = df['algorithm'].replace({'SMAlegAATr': 'SMAlegAAATr'})

    # Calculate the scores
    defect_scores, coop_scores = {}, {}
    minimax_val, lowest_reward = 3.3769999999999993, 0
    opp_types = df['opponent_type'].unique()
    for agent in NAMES:
        for opp_type in opp_types:
            df_filtered = df[(df['algorithm'] == agent) & (df['opponent_type'] == opp_type)]
            rewards, reward_sums = df_filtered['agent_final_reward'].to_list(), df_filtered['reward_sum'].to_list()
            assert len(rewards) == len(reward_sums)
            for i in range(len(rewards)):
                if opp_type in ['greedyplannerhare', 'greedyprobhare', 'greedyhare', 'probdesthare']:
                    comparison = 3.3769999999999993
                    regret = (comparison - lowest_reward) - (rewards[i] - lowest_reward)
                    val = 1 - max((regret / (comparison - lowest_reward)), 0)
                    val = max(val, 0)
                    val = min(val, 1)
                    assert 0 <= val <= 1
                    defect_scores[agent] = defect_scores.get(agent, []) + [val]

                elif opp_type in ['greedystag', 'modellerstag', 'teamawarestag', 'greedyplannerstag', 'selfplay']:
                    comparison = 13.760833333333334
                    avg_reward = reward_sums[i] / 3
                    regret = (comparison - minimax_val) - (avg_reward - minimax_val)
                    val = 1 - min((regret / (comparison - minimax_val)), 1)
                    val = max(val, 0)
                    val = min(val, 1)
                    assert 0 <= val <= 1
                    coop_scores[agent] = coop_scores.get(agent, []) + [val]

                else:
                    raise Exception(f'{opp_type} is not a defined opponent type')

    print('PURSUIT:')
    scores = {}
    for agent in NAMES:
        d_scores, c_scores = defect_scores[agent], coop_scores[agent]
        c_scores = c_scores[:len(d_scores)]
        assert len(d_scores) == len(c_scores)
        a_scores = [min(d_scores[i], c_scores[i]) for i in range(len(d_scores))]
        d, c, a = np.array(d_scores).mean(), np.array(c_scores).mean(), np.array(a_scores).mean()
        scores[agent] = {}
        scores[agent]['d'] = d_scores
        scores[agent]['c'] = c_scores
        scores[agent]['a'] = a_scores

        print(f'{agent}: d = {d}, c = {c}, a = {a}')

    for cond in ['d', 'c', 'a']:
        avgs, ses = [], []
        for alg in NAMES:
            alg_data = scores[alg][cond]
            avgs.append(np.mean(alg_data))
            ses.append(np.std(alg_data, ddof=1) / np.sqrt(len(alg_data)))

        plt.figure(figsize=(10, 3))
        plt.grid()
        plt.bar(NAMES, avgs, yerr=ses, capsize=5, color=COLORS)
        plt.xlabel('Algorithm', fontsize=18, fontweight='bold')
        plt.ylabel('Score', fontsize=18, fontweight='bold')
        plt.savefig(f'../analysis/adaptability_plots/pursuit_{cond}.png', bbox_inches='tight')
        plt.clf()


def rg_scores(game_name: str) -> None:
    n_epochs, n_rounds = 50, 50
    defect_scores, coop_scores, adapt_scores = {}, {}, {}

    if game_name == 'chicken':
        minimax_val, lowest_reward = -1 * n_rounds, -5 * n_rounds
        defect_comparison, coop_comparison = -1 * n_rounds, 1 * n_rounds

    elif game_name == 'coordination':
        minimax_val, lowest_reward = 0 * n_rounds, 0 * n_rounds
        defect_comparison, coop_comparison = 2 * n_rounds, 2 * n_rounds

    elif game_name == 'prisoners_dilemma':
        minimax_val, lowest_reward = -1 * n_rounds, -3 * n_rounds
        defect_comparison, coop_comparison = -1 * n_rounds, 3 * n_rounds

    else:
        raise Exception(f'{game_name} is not a valid game')

    for agent in NAMES:
        name_for_file = 'smalegaatr' if agent == 'SMAlegAAATr' else 'algaater'
        sp_column = 'SMAlegAATr' if agent == 'SMAlegAAATr' else 'Algaater'
        file_adj = '_auto' if agent == 'AlegAAATr' else ('_auto_tuned' if agent == 'AlegAAATTr' else '')
        df_train = pd.read_csv(f'../analysis/{game_name}_game/{name_for_file}_full{file_adj}.csv')
        df_self_play = pd.read_csv(f'../analysis/{game_name}_game/{name_for_file}_self_play{file_adj}.csv')
        for i in range(n_epochs):
            defect_rewards = list(df_train.loc[df_train.index[i], ['BullyOpp', 'BullyPunishOpp']])
            coop_rewards = [df_train.loc[df_train.index[i], 'CoopPunishOpp'], df_self_play.loc[df_self_play.index[i], sp_column]]

            # Defect score
            avg_d_reward = sum(defect_rewards) / len(defect_rewards)
            regret = (defect_comparison - lowest_reward) - (avg_d_reward - lowest_reward)
            d = 1 - max((regret / (defect_comparison - lowest_reward)), 0)
            d = max(d, 0)
            d = min(d, 1)
            assert 0 <= d <= 1
            defect_scores[agent] = defect_scores.get(agent, []) + [d]

            # Coop score
            avg_c_reward = sum(coop_rewards) / len(coop_rewards)
            regret = (coop_comparison - minimax_val) - (avg_c_reward - minimax_val)
            c = 1 - min((regret / (coop_comparison - minimax_val)), 1)
            c = max(c, 0)
            c = min(c, 1)
            assert 0 <= c <= 1
            coop_scores[agent] = coop_scores.get(agent, []) + [c]

            # Adapt score
            adapt_scores[agent] = adapt_scores.get(agent, []) + [min(d, c)]

    print(f'{game_name.upper()}:')
    scores = {}
    for agent in NAMES:
        d_scores, c_scores, a_scores = defect_scores[agent], coop_scores[agent], adapt_scores[agent]
        d, c, a = np.array(d_scores).mean(), np.array(c_scores).mean(), np.array(a_scores).mean()
        scores[agent] = {}
        scores[agent]['d'] = d_scores
        scores[agent]['c'] = c_scores
        scores[agent]['a'] = a_scores

        print(f'{agent}: d = {d}, c = {c}, a = {a}')

    for cond in ['d', 'c', 'a']:
        avgs, ses = [], []
        for alg in NAMES:
            alg_data = scores[alg][cond]
            avgs.append(np.mean(alg_data))
            ses.append(np.std(alg_data, ddof=1) / np.sqrt(len(alg_data)))

        plt.figure(figsize=(10, 3))
        plt.grid()
        plt.bar(NAMES, avgs, yerr=ses, capsize=5, color=COLORS)
        plt.xlabel('Algorithm', fontsize=18, fontweight='bold')
        plt.ylabel('Score', fontsize=18, fontweight='bold')
        plt.savefig(f'../analysis/adaptability_plots/{game_name}_{cond}.png', bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    # forex_scores()
    rg_scores('prisoners_dilemma')
    # pursuit_scores()
