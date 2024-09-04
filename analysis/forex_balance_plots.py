import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def create_plots() -> None:
    def _create_final_balance_bar_graphs() -> None:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[1:]:
                    pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'
                    file_list = os.listdir('../analysis/forex_results/')
                    filtered_file_list = [file_name for file_name in file_list if
                                          (pair_time_year_str in file_name and 'final_balances' in file_name)]

                    final_balances, strategy_names, max_len = [], [], 0

                    for file_name in filtered_file_list:
                        strategy_name = file_name.split('_')[0]
                        final_balance = pickle.load(open(f'../analysis/forex_results/{file_name}', 'rb'))

                        strategy_names.append(strategy_name)
                        final_balances.append(final_balance)

                    # Bar graph containing final balances for each strategy
                    bar_colors = ['blue', 'green']
                    plt.grid()
                    plt.bar(strategy_names, final_balances, color=bar_colors)
                    plt.axhline(y=10000, color='black', linestyle='--', linewidth=2)
                    plt.xlabel('Strategy')
                    plt.ylabel('Final Account Balance')
                    plt.savefig(f'../analysis/forex_figures/{pair_time_year_str}_final_balances.png', bbox_inches='tight')
                    plt.clf()

                    # Export the final balances as a csv
                    df = pd.DataFrame([final_balances], columns=strategy_names)
                    df.to_csv(f'../analysis/forex_results/final_balances_csv/{pair_time_year_str}_final_balances.csv')

    # Create bar graphs of the final balances
    _create_final_balance_bar_graphs()


if __name__ == "__main__":
    create_plots()
