from copy import deepcopy
from pandas import DataFrame
import pickle
from forex.market_proxy.trade import Trade, TradeType
from forex.market_proxy.market_simulation_results import MarketSimulationResults
from typing import Optional


# Abstract strategies class that each specific strategies must implement
class Strategy:
    def __init__(self, starting_idx: int, percent_to_risk: float, name: str) -> None:
        self.starting_idx = starting_idx
        self.percent_to_risk = percent_to_risk
        self.name = name

    def print_parameters(self) -> None:
        for name, val in self.__dict__.items():
            print(f'{name}: {val}')

    def trade_finished(self, net_profit: float) -> None:
        pass

    # Loads in the best parameter values that were estimated by the genetic algorithm
    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        pair_time_frame_year_str = f'{currency_pair}_{time_frame}_{year}'
        best_params_dictionary = pickle.load(
            open(f'../genetics/best_genome_features/{self.name}_{pair_time_frame_year_str}_features.pickle', 'rb'))

        for attribute_name, val in best_params_dictionary.items():
            self.__setattr__(attribute_name, val)

            if attribute_name == 'lookback' or attribute_name == 'n_in_a_row':
                self.__setattr__('starting_idx', val)

    # Each strategies has unique rules to determine if a trade should be placed
    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        pass

    # A strategies might have rules to move the stop loss if the market moves in the trade's favor
    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        trade_copy = deepcopy(trade)
        curr_bid_high, curr_ask_low = market_data.loc[market_data.index[curr_idx], ['Bid_High', 'Ask_Low']]

        # Move the stop loss on a buy if the market is in our favor
        if trade_copy.trade_type == TradeType.BUY and curr_bid_high - trade_copy.pips_risked > trade_copy.stop_loss:
            trade_copy.stop_loss = curr_bid_high - trade_copy.pips_risked

        elif trade_copy.trade_type == TradeType.SELL and curr_ask_low + trade_copy.pips_risked < trade_copy.stop_loss:
            trade_copy.stop_loss = curr_ask_low + trade_copy.pips_risked

        return trade_copy

    # A strategies might want to close part of the trade and move the stop loss after the market has moved a certain
    # amount in the trade's favor
    def close_part_of_trade(self, curr_idx: int, market_data: DataFrame, trade: Trade,
                            simulation_results: MarketSimulationResults, currency_pair: str) -> Optional[Trade]:
        trade_copy = deepcopy(trade)
        return trade_copy
