from copy import deepcopy
from forex.market_proxy.market_simulation_results import MarketSimulationResults
from forex.market_proxy.trade import Trade, TradeType
import numpy as np
from pandas import DataFrame
import pickle
from forex.strategies.bar_movement import BarMovement
from forex.strategies.beep_boop import BeepBoop
from forex.strategies.bollinger_bands import BollingerBands
from forex.strategies.choc import Choc
from forex.strategies.keltner_channels import KeltnerChannels
from forex.strategies.ma_crossover import MACrossover
from forex.strategies.macd import MACD
from forex.strategies.macd_key_level import MACDKeyLevel
from forex.strategies.macd_stochastic import MACDStochastic
from forex.strategies.psar import PSAR
from forex.strategies.rsi import RSI
from forex.strategies.squeeze_pro import SqueezePro
from forex.strategies.stochastic import Stochastic
from forex.strategies.strategy import Strategy
from forex.strategies.supertrend import Supertrend
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from typing import Optional, Tuple
from utils.utils import FOREX_E_DESCRIPTION, PAD_VAL


@keras.saving.register_keras_serializable()
class AutoAATHead(Model):
    def __init__(self, aat_dim: int = 100, state_dim: int = 300) -> None:
        super(AutoAATHead, self).__init__()
        self.aat_dim = aat_dim
        self.state_dim = state_dim

        self.dense_aat_1 = Dense(self.aat_dim, activation='relu')
        self.dense_aat_2 = Dense(32, activation='relu')

        # self.dense_state_1 = Dense(self.state_dim, activation='relu')
        # self.dense_state_2 = Dense(32, activation='relu')
        #
        # self.dense_combined_1 = Dense(32, activation='relu')
        # self.dense_combined_2 = Dense(32, activation='relu')

        self.output_layer = Dense(14, activation='linear')

    def get_config(self):
        return {'state_dim': self.state_dim, 'aat_dim': self.aat_dim}

    def call(self, aat_state, return_transformed_state: bool = False) -> tf.Tensor:
        x_aat = self.dense_aat_1(aat_state)
        x_aat = x_aat + aat_state
        x_aat = self.dense_aat_2(x_aat)

        # x_state = self.dense_state_1(state)
        # x_state = x_state + state
        # x_state = self.dense_state_2(x_state)
        #
        # x = x_aat + x_state
        # x = x + self.dense_combined_1(x)
        # x = self.dense_combined_2(x)
        #
        # if return_transformed_state:
        #     return x
        #
        # return self.output_layer(x)

        if return_transformed_state:
            return x_aat

        return self.output_layer(x_aat)


@keras.saving.register_keras_serializable()
class AutoAATWithHead(Model):
    def __init__(self, auto_aat_model: Model, auto_aat_head: Model) -> None:
        super(AutoAATWithHead, self).__init__()
        self.auto_aat_model = auto_aat_model
        self.auto_aat_head = auto_aat_head

    def get_config(self):
        return {'auto_aat_model': keras.saving.serialize_keras_object(self.auto_aat_model),
                'auto_aat_head': keras.saving.serialize_keras_object(self.auto_aat_head)}

    @classmethod
    def from_config(cls, config):
        auto_aat_model_config = config.pop('auto_aat_model')
        auto_aat_model = keras.saving.deserialize_keras_object(auto_aat_model_config)
        config['auto_aat_model'] = auto_aat_model

        auto_aat_head_config = config.pop('auto_aat_head')
        auto_aat_head = keras.saving.deserialize_keras_object(auto_aat_head_config)
        config['auto_aat_head'] = auto_aat_head

        return cls(**config)

    def call(self, data_tup: Tuple[np.array, np.array, np.array], training: bool = False,
             return_transformed_state: bool = False) -> tf.Tensor:
        aat_vec = self.auto_aat_model(data_tup, training=training)

        return self.auto_aat_head(aat_vec, return_transformed_state=return_transformed_state)


class QAlegAATr(Strategy):
    def __init__(self, name: str = 'QAlegAATr', starting_idx: int = 2, percent_to_risk: float = 0.02,
                 invert: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, name)
        self.generators = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                           MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                           Supertrend()]
        self.invert = invert
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0
        self.generator_in_use_name = None
        self.tracked_vector = None

    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'

        self.model = load_model(f'../aat/auto_aat_tuned/{pair_time_year_str}_tuned_model.keras')
        self.scaler = pickle.load(open(f'../aat/auto_aat_tuned/{pair_time_year_str}_tuned_scaler.pickle', 'rb'))

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for generator in self.generators:
            self.min_idx = max(self.min_idx, generator.starting_idx)

        # Safety check
        if curr_idx < self.min_idx:
            return None

        # Raw state
        g_description = 'Generator that represents a trading strategy'
        state_input = strategy_data.loc[
            strategy_data.index[curr_idx - 1], ['Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume',
                                                'atr', 'lower_atr_band', 'upper_atr_band', 'ema200',
                                                'ema100', 'ema50', 'smma200', 'smma100', 'smma50',
                                                'bid_pips_down', 'bid_pips_up', 'ask_pips_down',
                                                'ask_pips_up', 'rsi', 'rsi_sma', 'adx', 'chop', 'vo',
                                                'rsi_up', 'adx_large', 'chop_small', 'vo_positive',
                                                'squeeze_on', 'macd', 'macdsignal', 'macdhist', 'beep_boop',
                                                'support_fractal', 'resistance_fractal', 'sar', 'lower_kc',
                                                'upper_kc', 'lower_bb', 'upper_bb', 'qqe_up', 'qqe_down',
                                                'qqe_val', 'supertrend', 'supertrend_ub', 'supertrend_lb',
                                                'slowk', 'slowd', 'slowk_rsi', 'slowd_rsi', 'n_macd',
                                                'n_macdsignal', 'impulse_macd', 'impulse_macdsignal']]
        state_input = list(state_input)
        state_input += [PAD_VAL] * (300 - len(state_input))
        state_input = self.scaler.transform(np.array(state_input, dtype=float).reshape(1, -1))

        # Make predictions
        q_values = self.model((np.array(g_description).reshape(1, -1),
                               np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                               state_input)).numpy()
        self.tracked_vector = self.model((np.array(g_description).reshape(1, -1),
                                          np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                          state_input),
                                         return_transformed_state=True).numpy().reshape(-1, )

        best_generator_idx = np.argmax(q_values)
        trade = self.generators[best_generator_idx].place_trade(curr_idx, strategy_data, currency_pair,
                                                                account_balance)
        self.generator_in_use_name = self.generators[best_generator_idx].name

        if trade is not None:
            curr_ao, curr_bo = strategy_data.loc[strategy_data.index[curr_idx], ['Ask_Open', 'Bid_Open']]

            return self._invert(trade, curr_ao, curr_bo)

        return None

    def _invert(self, trade: Trade, curr_ao: float, curr_bo: float) -> Trade:
        if self.invert:
            trade_copy = deepcopy(trade)

            if trade.trade_type == TradeType.BUY:
                trade_copy.trade_type = TradeType.SELL
                trade_copy.open_price = curr_bo
                trade_copy.stop_loss = curr_bo + trade.pips_risked

                if trade.stop_gain is not None:
                    stop_gain_pips = abs(trade.stop_gain - trade.open_price)
                    trade_copy.stop_gain = curr_bo - stop_gain_pips

            else:
                trade_copy.trade_type = TradeType.BUY
                trade_copy.open_price = curr_ao
                trade_copy.stop_loss = curr_ao - trade.pips_risked

                if trade.stop_gain is not None:
                    stop_gain_pips = abs(trade.stop_gain - trade.open_price)
                    trade_copy.stop_gain = curr_ao + stop_gain_pips

            return trade_copy

        else:
            return trade

    def move_stop_loss(self, curr_idx: int, market_data: DataFrame, trade: Trade) -> Trade:
        if self.use_tsl:
            return super().move_stop_loss(curr_idx, market_data, trade)

        else:
            return trade

    def close_part_of_trade(self, curr_idx: int, market_data: DataFrame, trade: Trade,
                            simulation_results: MarketSimulationResults, currency_pair: str) -> Optional[Trade]:
        if self.close_trade_incrementally:
            return super().close_part_of_trade(curr_idx, market_data, trade, simulation_results, currency_pair)

        else:
            return trade
