import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES


def crunch_numbers() -> None:
    def total_profit() -> None:
        all_profits, recent_two_profits, m30_profits, h1_profits, h4_profits = {}, {}, {}, {}, {}
        all_profit_sums, m30_profit_sums, h1_profit_sums, h4_profit_sums = {}, {}, {}, {}
        eur_usd_profits, usd_jpy_profits, gbp_chf_profits = {}, {}, {}
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
                all_profits[strategy] = all_profits.get(strategy, []) + [profit]
                all_profit_sums[strategy] = all_profit_sums.get(strategy, 0) + profit

                if 'M30' in file_name:
                    m30_profits[strategy] = m30_profits.get(strategy, []) + [profit]
                    m30_profit_sums[strategy] = m30_profit_sums.get(strategy, 0) + profit

                elif 'H1' in file_name:
                    h1_profits[strategy] = h1_profits.get(strategy, []) + [profit]
                    h1_profit_sums[strategy] = h1_profit_sums.get(strategy, 0) + profit

                else:
                    h4_profits[strategy] = h4_profits.get(strategy, []) + [profit]
                    h4_profit_sums[strategy] = h4_profit_sums.get(strategy, 0) + profit

                if '2021' in file_name or '2022' in file_name:
                    recent_two_profits[strategy] = recent_two_profits.get(strategy, []) + [profit]

                if 'Eur_Usd' in file_name:
                    eur_usd_profits[strategy] = eur_usd_profits.get(strategy, []) + [profit]

                elif 'Usd_Jpy' in file_name:
                    usd_jpy_profits[strategy] = usd_jpy_profits.get(strategy, []) + [profit]

                else:
                    gbp_chf_profits[strategy] = gbp_chf_profits.get(strategy, []) + [profit]

            for pair_time_frame in results_by_pair_time_frame.keys():
                if pair_time_frame in file_name:
                    for strategy in df.columns[1:]:
                        profit = df.loc[df.index[0], strategy] - 10000
                        results_by_pair_time_frame[pair_time_frame][strategy] = results_by_pair_time_frame[
                                                                                    pair_time_frame].get(strategy,
                                                                                                         []) + [profit]

        # Create tuples for each strategy and profit sum
        profit_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in all_profits.items()]
        m30_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in m30_profits.items()]
        h1_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in h1_profits.items()]
        h4_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in h4_profits.items()]
        eur_usd_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in eur_usd_profits.items()]
        usd_jpy_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in usd_jpy_profits.items()]
        gbp_chf_with_names = [(strategy, np.array(profit).mean()) for strategy, profit in gbp_chf_profits.items()]

        # Sort the results so that the most profitable results are first
        profit_with_names.sort(key=lambda x: x[1], reverse=True)
        m30_with_names.sort(key=lambda x: x[1], reverse=True)
        h1_with_names.sort(key=lambda x: x[1], reverse=True)
        h4_with_names.sort(key=lambda x: x[1], reverse=True)

        # Print the profit averages and print latex table (for the paper)
        total_avg, total_se = 0, 0
        latex_data, latex_headers = {}, ['Strategy', 'Overall']
        print('PROFIT AVERAGES ACROSS EVERY CATEGORY')

        for strategy, avg in profit_with_names:
            print(f'{strategy}\'s average profit: {avg}')
            total_avg += avg
            latex_data[strategy] = [avg]
            profits = np.array(all_profits[strategy])
            se = profits.std() / len(profits) ** 0.5
            print(f'{strategy}\'s standard error: {round(se)}')
            total_se += se

        print(f'AVERAGE PROFIT AVG: {total_avg / len(profit_with_names)}')
        print(f'AVERAGE PROFIT SE: {total_se / len(profit_with_names)}\n')

        for time_frame in TIME_FRAMES:
            latex_headers.append(time_frame)
            total_avg, curr_row = 0, []
            sum_to_print = m30_with_names if time_frame == 'M30' else (
                h1_with_names if time_frame == 'H1' else h4_with_names)

            print(f'PROFIT AVERAGES FOR {time_frame}:')

            for strategy, avg in sum_to_print:
                print(f'{strategy}\'s average profit: {avg}')
                total_avg += avg
                latex_data[strategy] += [avg]

            print(f'AVERAGE PROFIT AVG: {total_avg / len(sum_to_print)}\n')

        for currency_pair in CURRENCY_PAIRS:
            latex_headers.append(currency_pair)
            total_avg, curr_row = 0, []
            sum_to_print = eur_usd_with_names if currency_pair == 'Eur_Usd' else (
                usd_jpy_with_names if currency_pair == 'Usd_Jpy' else gbp_chf_with_names)

            print(f'PROFIT AVERAGES FOR {currency_pair}:')

            for strategy, avg in sum_to_print:
                print(f'{strategy}\'s average profit: {avg}')
                total_avg += avg
                latex_data[strategy] += [avg]

            print(f'AVERAGE PROFIT AVG: {total_avg / len(sum_to_print)}\n')

        latex_df = pd.DataFrame(latex_data).T.reset_index()
        latex_df.columns = latex_headers
        latex_headers.append(latex_headers.pop(1))
        latex_df = latex_df[latex_headers]

        # print(latex_df.to_latex(index=False, float_format=lambda x: '{:.0f}'.format(round(x))))
        print(latex_df)

        # Print the profit results for each currency and time frame pair
        for pair_time_frame in results_by_pair_time_frame.keys():
            print(f'PAIR-TIME RESULTS FOR {pair_time_frame}:')

            for strategy, profits in results_by_pair_time_frame[pair_time_frame].items():
                profits_array = np.array(profits)

                print(f'{strategy}\'s avg profit: {profits_array.mean()}, med profit: {np.median(profits_array)}, '
                      f'profit std: {profits_array.std()}, min: {profits_array.min()}, max: {profits_array.max()}, '
                      f'profit sum: {profits_array.sum()}')

            print()

        # Print the profit results for each time frame
        for time_frame in TIME_FRAMES:
            profits_to_print = m30_profits if time_frame == 'M30' else (
                h1_profits if time_frame == 'H1' else h4_profits)

            print(f'PROFIT RESULTS FOR {time_frame}:')

            for strategy, profits in profits_to_print.items():
                profits_array = np.array(profits)

                print(f'{strategy}\'s avg profit: {profits_array.mean()}, med profit: {np.median(profits_array)}, '
                      f'profit std: {profits_array.std()}, min: {profits_array.min()}, max: {profits_array.max()}')

            print()

        all_profs, names = [], []

        for strategy, profits in sorted(all_profits.items(), key=lambda item: sum(item[1]) / len(item[1]),
                                        reverse=True):
            profits_array = np.array(profits)
            avg, sd = profits_array.mean(), profits_array.std()
            print(f'{strategy}\'s avg profit: {avg}, med profit: {np.median(profits_array)}, '
                  f'profit std: {sd}, min: {profits_array.min()}, max: {profits_array.max()}')

            all_profs.append(profits_array)
            names.append(strategy)

        avgs = [arry.mean() for arry in all_profs]
        standard_errors = [arry.std() / len(arry) ** 0.5 for arry in all_profs]
        bar_colors = ['blue', 'green', 'red', 'orange']
        plt.grid()
        plt.bar(names, avgs, yerr=standard_errors, color=bar_colors)
        plt.xlabel('Strategy')
        plt.ylabel('Amount ($)')
        plt.savefig(f'../analysis/forex_figures/avg_profit_amounts.png', bbox_inches='tight')
        plt.clf()

    def profitable_ratios() -> None:
        # Print out the profit ratio of the test sets
        directory = '../analysis/forex_results/ratios_csv/'
        file_list = os.listdir(directory)
        ratios, ratios_with_names = [], []

        for file_name in file_list:
            df = pd.read_csv(f'{directory}{file_name}')

            for strategy in df.columns[1:]:
                profit_ratio = df.loc[df.index[0], strategy]
                ratios.append(profit_ratio)
                ratios_with_names.append((strategy, profit_ratio))

        ratios_with_names.sort(key=lambda x: x[1], reverse=True)

        for strategy, ratio in ratios_with_names:
            print(f'{strategy}\'s profitable ratio: {ratio}')

        print(f'\nAverage profitable ratio: {np.array(ratios).mean()}, med profitable ratio: {np.median(ratios)}')

    # Calculate profit sums
    total_profit()

    print()

    # Print out profit ratios
    profitable_ratios()


if __name__ == "__main__":
    crunch_numbers()
