import matplotlib.pyplot as plt
import numpy as np
import pickle
from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def process_alegaatr_metrics() -> None:
    for auto_aat in [True, False]:
        file_adjustment = '_auto' if auto_aat else ''
        predictions_when_wrong, trade_values_when_wrong, predictions_when_correct, trade_values_when_correct = \
            [], [], [], []

        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[2:]:
                    file_path = f'../analysis/forex_results/alegaatr_metrics/{currency_pair}_{time_frame}_{year}'
                    predictions_when_wrong += pickle.load(open(f'{file_path}_predictions_when_wrong{file_adjustment}.pickle', 'rb'))
                    trade_values_when_wrong += pickle.load(open(f'{file_path}_trade_values_when_wrong{file_adjustment}.pickle', 'rb'))
                    predictions_when_correct += pickle.load(open(f'{file_path}_predictions_when_correct{file_adjustment}.pickle', 'rb'))
                    trade_values_when_correct += pickle.load(
                        open(f'{file_path}_trade_values_when_correct.pickle', 'rb'))

        predictions_when_wrong_clean = [val for val in predictions_when_wrong if val != np.inf]
        predictions_when_correct_clean = [val for val in predictions_when_correct if val != np.inf]

        n_bins = int(0.25 * len(predictions_when_wrong_clean))

        plt.grid()
        plt.hist(predictions_when_wrong_clean, bins=n_bins, alpha=0.75, label='Predictions', color='red')
        plt.hist(trade_values_when_wrong, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.savefig(f'../analysis/forex_figures/incorrect{file_adjustment}.png', bbox_inches='tight')
        plt.clf()

        n_bins = int(0.25 * len(predictions_when_correct_clean))

        plt.grid()
        plt.hist(predictions_when_correct_clean, bins=n_bins, alpha=0.75, label='Predictions', color='blue')
        plt.hist(trade_values_when_correct, bins=n_bins, alpha=0.5, label='Trade Amounts', color='green')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.savefig(f'../analysis/forex_figures/correct{file_adjustment}.png', bbox_inches='tight')
        plt.clf()

        plt.grid()
        plt.hist(predictions_when_correct_clean, bins=n_bins, alpha=0.75, label='Correct Predictions', color='green')
        plt.hist(predictions_when_wrong_clean, bins=n_bins, alpha=0.5, label='Incorrect Predictions', color='red')
        plt.xlabel('USD Amounts')
        plt.ylabel('Counts')
        plt.legend(loc='best')
        plt.savefig(f'../analysis/forex_figures/correct_incorrect{file_adjustment}.png', bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    process_alegaatr_metrics()
