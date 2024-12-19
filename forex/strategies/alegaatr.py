from forex.aat.assumptions import Assumptions
from collections import deque
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
from tensorflow.keras.models import load_model
from typing import Optional, List
from utils.utils import FOREX_E_DESCRIPTION,FOREX_G_DESCRIPTIONS, NETWORK_NAME, PAD_VAL


class AlegAATr(Strategy):
    def __init__(self, name: str = 'AlegAATr', starting_idx: int = 2, percent_to_risk: float = 0.02, min_num_predictions: int = 0,
                 use_single_selection: bool = True, invert: bool = False, min_neighbors: int = 1,
                 max_neighbors: int = 100000, genetic: bool = False, lmbda: float = 0.0,
                 optimistic_start: bool = False, auto_aat: bool = False, auto_aat_tuned: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, name)
        self.generators = [BarMovement(), BeepBoop(), BollingerBands(), Choc(), KeltnerChannels(), MACrossover(),
                           MACD(), MACDKeyLevel(), MACDStochastic(), PSAR(), RSI(), SqueezePro(), Stochastic(),
                           Supertrend()]
        self.auto_aat = auto_aat
        self.auto_aat_tuned = auto_aat_tuned
        self.models, self.correction_terms = {}, {}
        self.min_num_predictions, self.use_single_selection, self.invert, self.min_neighbors, self.max_neighbors = \
            min_num_predictions, use_single_selection, invert, min_neighbors, max_neighbors
        self.use_tsl, self.close_trade_incrementally = False, False
        self.min_idx = 0
        self.prev_prediction = None
        self.predictions_when_wrong, self.trade_values_when_wrong = [], []
        self.predictions_when_correct, self.trade_values_when_correct = [], []
        self.genetic, self.lmbda, self.optimistic_start = genetic, lmbda, optimistic_start
        self.empirical_rewards, self.time_since_used = {}, {}
        self.generator_in_use_name = None

        for generator in self.generators:
            generator_name = generator.name
            self.empirical_rewards[generator_name] = deque(maxlen=5)
            self.time_since_used[generator_name] = 0

        if self.auto_aat:
            self.assumption_pred_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
            self.state_scaler = pickle.load(open(f'../../networks/scalers/{NETWORK_NAME}_state_scaler.pickle', 'rb'))
            assert self.state_scaler._scaler is not None

        self.tracked_vector = None

    def print_parameters(self) -> None:
        print(f'min_num_predictions: {self.min_num_predictions}')
        print(f'use_single_selection: {self.use_single_selection}')
        print(f'lmbda: {self.lmbda}')
        print(f'optimistic_start: {self.optimistic_start}')

    def _clear_metric_tracking_vars(self) -> None:
        self.prev_prediction = None
        self.predictions_when_wrong.clear()
        self.trade_values_when_wrong.clear()
        self.predictions_when_correct.clear()
        self.trade_values_when_correct.clear()

    def update_metric_tracking_vars(self, trade_value: float) -> None:
        # Win
        if trade_value > 0:
            self.predictions_when_correct.append(self.prev_prediction)
            self.trade_values_when_correct.append(trade_value)

        # Loss
        else:
            self.predictions_when_wrong.append(self.prev_prediction)
            self.trade_values_when_wrong.append(trade_value)

        self.prev_prediction = None

    def save_metric_tracking_vars(self, currency_pair: str, time_frame: str, year: int) -> None:
        file_path = f'../../analysis/forex_results/alegaatr_metrics/{currency_pair}_{time_frame}_{year}'
        file_adjustment = '_auto' if self.auto_aat else ''

        with open(f'{file_path}_predictions_when_wrong{file_adjustment}.pickle', 'wb') as f:
            pickle.dump(self.predictions_when_wrong, f)

        with open(f'{file_path}_trade_values_when_wrong{file_adjustment}.pickle', 'wb') as f:
            pickle.dump(self.trade_values_when_wrong, f)

        with open(f'{file_path}_predictions_when_correct{file_adjustment}.pickle', 'wb') as f:
            pickle.dump(self.predictions_when_correct, f)

        with open(f'{file_path}_trade_values_when_correct{file_adjustment}.pickle', 'wb') as f:
            pickle.dump(self.trade_values_when_correct, f)

        self._clear_metric_tracking_vars()

    def trade_finished(self, net_profit: float) -> None:
        self.empirical_rewards[self.generator_in_use_name].append(net_profit)

    def load_best_parameters(self, currency_pair: str, time_frame: str, year: int) -> None:
        if self.auto_aat_tuned:
            pair_time_year_str = f'{currency_pair}_{time_frame}_{year}'

            self.model = load_model(f'../aat/auto_aat_tuned/{pair_time_year_str}_tuned_model.keras')
            self.scaler = pickle.load(open(f'../aat/auto_aat_tuned/{pair_time_year_str}_tuned_scaler.pickle', 'rb'))

        file_adjustment = '_auto' if self.auto_aat else ('autotuned' if self.auto_aat_tuned else '')

        for generator in self.generators:
            try:
                # Load the KNN model and correction terms for the generator on the given currency pair and time frame
                strategy_name = generator.name
                name_pair_time_year_str = f'{strategy_name}_{currency_pair}_{time_frame}_{year}'
                correction_terms_file_name = \
                    f'../aat/training_data/{name_pair_time_year_str}_aat_correction_terms{file_adjustment}.pickle'
                knn_file_name = f'../aat/training_data/{name_pair_time_year_str}_aat_knn{file_adjustment}.pickle'
                self.models[strategy_name] = pickle.load(open(knn_file_name, 'rb'))
                self.correction_terms[strategy_name] = pickle.load(open(correction_terms_file_name, 'rb'))

            except:
                continue

        if self.lmbda == 1.0:
            self.optimistic_start = True

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        for generator in self.generators:
            self.min_idx = max(self.min_idx, generator.starting_idx)

        # Safety check
        if curr_idx < self.min_idx:
            return None

        vectors = []
        tracked_vectors = []

        if self.auto_aat:
            assert self.assumption_pred_model is not None and self.state_scaler is not None

            for gen in self.generators:
                g_description = FOREX_G_DESCRIPTIONS[gen.name]
                state_input = strategy_data.loc[strategy_data.index[curr_idx - 1], ['Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume',
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
                state_input_scaled = self.state_scaler.scale(np.array(state_input, dtype=float).reshape(1, -1))
                assumption_preds = self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                                                               np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                                               state_input_scaled)).numpy()
                assumption_preds = list(assumption_preds[0, :34])
                assumption_preds.append(0.0)
                new_assumptions = Assumptions(strategy_data, curr_idx, currency_pair, 0.0, calculate=False)
                new_assumptions.set_vals(*assumption_preds)

                tracked_vectors.append(self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                                                                  np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                                                  state_input_scaled),
                                                                 return_transformed_state=True).numpy().reshape(-1, ))
                vectors.append(np.array(new_assumptions.create_aat_tuple()[:-1], dtype=float).reshape(1, -1))

        elif self.auto_aat_tuned:
            assert self.model is not None and self.scaler is not None

            for gen in self.generators:
                g_description = FOREX_G_DESCRIPTIONS[gen.name]
                state_input = strategy_data.loc[strategy_data.index[curr_idx - 1], ['Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume',
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
                state_input_scaled = self.scaler.transform(np.array(state_input, dtype=float).reshape(1, -1))
                assumption_preds = self.model.auto_aat_model((np.array(g_description).reshape(1, -1),
                                                         np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                                         state_input_scaled)).numpy()
                assumption_preds = list(assumption_preds[0, :34])
                assumption_preds.append(0.0)
                new_assumptions = Assumptions(strategy_data, curr_idx, currency_pair, 0.0, calculate=False)
                new_assumptions.set_vals(*assumption_preds)

                tracked_vectors.append(self.model.auto_aat_model((np.array(g_description).reshape(1, -1),
                                                                  np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                                                  state_input_scaled),
                                                                 return_transformed_state=True).numpy().reshape(-1, ))

                vectors.append(np.array(new_assumptions.create_aat_tuple()[:-1], dtype=float).reshape(1, -1))

        else:
            new_assumptions = Assumptions(strategy_data, curr_idx, currency_pair, 0.0)
            tracked_vectors.append(np.array(new_assumptions.create_aat_tuple()).reshape(-1, ))
            vectors.append(np.array(new_assumptions.create_aat_tuple()[:-1], dtype=float).reshape(1, -1))

        return self._single_selection(vectors, tracked_vectors, curr_idx, strategy_data, currency_pair, account_balance)

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

    def _single_selection(self, vectors: List[np.array], tracked_vectors: List[np.array], curr_idx: int, strategy_data: DataFrame, currency_pair: str,
                          account_balance: float) -> Optional[Trade]:
        best_trade_amount, n_profitable_predictions, best_generator_idx = -np.inf, 0, 0
        assert len(vectors) == len(tracked_vectors)

        for j in range(len(self.generators)):
            generator = self.generators[j]
            generator_name = generator.name
            x = vectors[j]

            if generator_name in self.models:
                prob = self.lmbda ** self.time_since_used[generator_name]
                use_empricial_avg = np.random.choice([1, 0], p=[prob, 1 - prob])
                empirical_rewards = self.empirical_rewards[generator_name]
                trade_amount_pred = None

                if use_empricial_avg and (len(empirical_rewards) > 0 or self.optimistic_start):
                    trade_amount_pred = np.array(empirical_rewards).mean() if len(empirical_rewards) > 0 else np.inf

                else:
                    knn_model, training_data = self.models[generator.name], self.correction_terms[generator.name]
                    n_neighbors = len(training_data)

                    if self.min_neighbors <= n_neighbors <= self.max_neighbors:
                        baseline = account_balance * self.percent_to_risk
                        correction_pred = knn_model.predict(x)[0]

                        trade_amount_pred = baseline * correction_pred

                if trade_amount_pred is not None:
                    n_profitable_predictions += 1 if trade_amount_pred > 0 else 0

                    if trade_amount_pred > best_trade_amount:
                        best_trade_amount, self.generator_in_use_name, best_generator_idx = \
                            trade_amount_pred, generator_name, j
                        self.use_tsl, self.close_trade_incrementally = \
                            generator.use_tsl, generator.close_trade_incrementally

        self.tracked_vector = tracked_vectors[best_generator_idx]

        for generator in self.generators:
            self.time_since_used[generator.name] += 1

        if n_profitable_predictions >= self.min_num_predictions:
            self.prev_prediction = best_trade_amount
            trade = self.generators[best_generator_idx].place_trade(curr_idx, strategy_data, currency_pair,
                                                                    account_balance)
            self.generator_in_use_name = self.generators[best_generator_idx].name

            if trade is not None:
                self.time_since_used[self.generator_in_use_name] = 0
                curr_ao, curr_bo = strategy_data.loc[strategy_data.index[curr_idx], ['Ask_Open', 'Bid_Open']]

                return self._invert(trade, curr_ao, curr_bo)

        return None

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
