from forex.data.data_loader import DataLoader
from forex.experiments.metrics_tracker import MetricsTracker
from forex.market_proxy.market_simulation_results import MarketSimulationResults
from forex.market_proxy.market_simulator import MarketSimulator
from forex.strategies.strategy import Strategy
from typing import Optional


class SimulationRunner(object):
    @staticmethod
    def run_simulation(strategy: Strategy, currency_pair: str, time_frame: str, year: int, optimize: bool,
                       train_aat: bool = False, metrics_tracker: Optional[MetricsTracker] = None,
                       auto_aat: bool = False, auto_aat_tuned: bool = False) -> MarketSimulationResults:
        market_data_raw = DataLoader.load_simulation_data(currency_pair, 'M5', optimize, year)
        strategy_data_raw = DataLoader.load_simulation_data(currency_pair, time_frame, optimize, year)

        # If we're running on test data (i.e. we're not optimizing anything), load in the strategy's "best" parameters
        try:
            if not optimize:
                strategy.load_best_parameters(currency_pair, time_frame, year - 1)

        except:
            # Case where there are no "best" parameters because they did not yield a positive net reward
            metrics_tracker.increment_profitable_testing(strategy.name, currency_pair, time_frame, year, True)
            metrics_tracker.update_final_balance(strategy.name, currency_pair, time_frame, year, 10000.0)

            return MarketSimulationResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        return MarketSimulator.run_simulation(strategy, market_data_raw, strategy_data_raw, currency_pair, time_frame,
                                              year, train_aat=train_aat, metrics_tracker=metrics_tracker,
                                              auto_aat=auto_aat, auto_aat_tuned=auto_aat_tuned)
