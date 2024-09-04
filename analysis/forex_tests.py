import os
import numpy as np
import pandas as pd
import pickle


def run_tests() -> None:
    def _extract_profitable_ratios() -> None:
        file_list = os.listdir('../analysis/forex_results/')
        filtered_file_list = [file_name for file_name in file_list if 'profitable_ratios' in file_name]

        ratios, strategy_names = [], []

        for file_name in filtered_file_list:
            strategy_name = file_name.split('_')[0]
            ratio = pickle.load(open(f'../analysis/forex_results/{file_name}', 'rb'))

            ratios.append(ratio)
            strategy_names.append(strategy_name)

        df = pd.DataFrame([ratios], columns=strategy_names)
        df.to_csv(f'../analysis/forex_results/ratios_csv/profitable_ratios.csv')

    def _effect_sizes_alegaatr() -> None:
        def _cohens_d(group1, group2):
            mean_diff = np.mean(group1) - np.mean(group2)
            pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

            return mean_diff / pooled_std

        profits_by_strategy = {}
        file_list = os.listdir('../analysis/forex_results/')
        filtered_file_list = [file_name for file_name in file_list if
                              ('final_balances' in file_name and 'csv' not in file_name)]

        for file_name in filtered_file_list:
            strategy_name = file_name.split('_')[0]
            profit = float(pickle.load(open(f'../analysis/forex_results/{file_name}', 'rb')) - 10000)
            profits_by_strategy[strategy_name] = profits_by_strategy.get(strategy_name, []) + [
                profit]

        alegaatr_profits = profits_by_strategy['AlegAATr']
        latex_df = []

        for name, profits in profits_by_strategy.items():
            if name == 'AlegAATr':
                continue

            d = _cohens_d(alegaatr_profits, profits)

            # print(f'Cohen\'s d AlegAATr vs. {name}: {d}')
            latex_df.append((f'AlegAATr vs. {name}', round(d, 3)))

        latex_df = pd.DataFrame(latex_df, columns=['Comparison', 'Effect Size'])
        # print(latex_df.to_latex(index=False))
        print(latex_df)

    # Extract the overall profitable ratios for each strategy
    _extract_profitable_ratios()

    # Cohen's d effect size for AlegAATr vs. all other strategies
    _effect_sizes_alegaatr()


if __name__ == "__main__":
    run_tests()
