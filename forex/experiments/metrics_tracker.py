import pickle
from forex.strategies.strategy import Strategy
from typing import List


class MetricsTracker:
    # For each strategy, want to keep track of:
    #   - Trade amounts (array of trade amounts)
    #       - With these, can calculate win rate and average trade amount
    #   - Account value over time (array of account values)
    #   - Final balance
    #   - Number of scenarios where profitable in testing / number of total testing scenarios
    def __init__(self) -> None:
        self.trade_amounts, self.account_values, self.final_balances = {}, {}, {}
        self.profitable_testing, self.profitable_ratios = {}, {}

    def update_strategy_metric_tracking_vars(self, strat: Strategy, trade_value: float) -> None:
        strat.update_metric_tracking_vars(trade_value)

    def increment_profitable_testing(self, strategy_name: str, currency_pair: str, time_frame: str, year: int,
                                     profitable: bool) -> None:
        strategy_pair_time_year_str = f'{strategy_name}_{currency_pair}_{time_frame}_{year}'
        amount = 1 if profitable else 0
        self.profitable_testing[strategy_pair_time_year_str] = self.profitable_testing.get(strategy_pair_time_year_str,
                                                                                           []) + [amount]

    def calculate_profitable_ratios(self, strategy_names: List[str]) -> None:
        for strategy_name in strategy_names:
            numerator, denominator = 0, 0

            for key, amounts in self.profitable_testing.items():
                if strategy_name in key:
                    numerator += sum(amounts)
                    denominator += len(amounts)

            self.profitable_ratios[strategy_name] = (numerator / denominator) if denominator > 0 else None

    def update_trade_amounts(self, strategy_name: str, currency_pair: str, time_frame: str, year: int,
                             trade_amount: float, account_balance: float) -> None:
        strategy_pair_time_year_str = f'{strategy_name}_{currency_pair}_{time_frame}_{year}'

        self.trade_amounts[strategy_pair_time_year_str] = self.trade_amounts.get(strategy_pair_time_year_str, []) + [
            trade_amount]

        # We can also update the account values here because account value changes with each trade
        self.account_values[strategy_pair_time_year_str] = self.account_values.get(strategy_pair_time_year_str, []) + [
            account_balance]

    def update_final_balance(self, strategy_name: str, currency_pair: str, time_frame: str, year: int,
                             final_balance: float) -> None:
        strategy_pair_time_year_str = f'{strategy_name}_{currency_pair}_{time_frame}_{year}'

        self.final_balances[strategy_pair_time_year_str] = self.final_balances.get(strategy_pair_time_year_str, []) + [
            final_balance]

    def save_data(self, strategy_names: List[str]) -> None:
        # Save the trade amounts
        for key, val in self.trade_amounts.items():
            file_location = f'../../analysis/forex_results/{key}_trade_amounts.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

        # Save the account values over time
        for key, val in self.account_values.items():
            file_location = f'../../analysis/forex_results/{key}_account_values.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

        # Save the final balances
        for key, val in self.final_balances.items():
            file_location = f'../../analysis/forex_results/{key}_final_balances.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(sum(val) / len(val), f)

        # Save the profitable testing set ratios
        self.calculate_profitable_ratios(strategy_names)

        for key, val in self.profitable_ratios.items():
            file_location = f'../../analysis/forex_results/{key}_profitable_ratios.pickle'

            with open(file_location, 'wb') as f:
                pickle.dump(val, f)

    def save_strategy_data(self, strat: Strategy, currency_pair: str, time_frame: str, year: int) -> None:
        strat.save_metric_tracking_vars(currency_pair, time_frame, year)
