from forex.aat.assumptions import Assumptions
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import load_model
from utils.utils import FOREX_E_DESCRIPTION, NETWORK_NAME, PAD_VAL


class AATTrainer:
    def __init__(self, currency_pair: str, strategy_name: str, time_frame: str, year: int, auto_aat: bool) -> None:
        self.currency_pair, self.strategy_name, self.time_frame, self.year = \
            currency_pair, strategy_name, time_frame, year
        self.auto_aat = auto_aat
        self.recent_tuple, self.training_data = None, []
        self.recent_state, self.states, self.trade_amounts = None, [], [] # For tuning auto AAT

        if self.auto_aat:
            self.assumption_pred_model = load_model(f'../../networks/models/{NETWORK_NAME}.keras')
            self.state_scaler = pickle.load(open(f'../../networks/scalers/{NETWORK_NAME}_state_scaler.pickle', 'rb'))
            assert self.state_scaler._scaler is not None

    # Adds a new AAT tuple to its training data
    def create_new_tuple(self, df: DataFrame, curr_idx: int, trade_amount: float) -> None:
        assert self.recent_tuple is None

        if self.auto_aat:
            assert self.assumption_pred_model is not None and self.state_scaler is not None
            g_description = 'Generator that represents a trading strategy'
            state_input = df.loc[df.index[curr_idx - 1], ['Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume',
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
            self.recent_state = state_input
            state_input_scaled = self.state_scaler.scale(np.array(state_input, dtype=float).reshape(1, -1))
            assumption_preds = self.assumption_pred_model((np.array(g_description).reshape(1, -1),
                                                     np.array(FOREX_E_DESCRIPTION).reshape(1, -1),
                                                     state_input_scaled)).numpy()
            assumption_preds = list(assumption_preds[0, :34])
            assumption_preds.append(trade_amount)
            new_assumptions = Assumptions(df, curr_idx, self.currency_pair, trade_amount, calculate=False)
            new_assumptions.set_vals(*assumption_preds)

        else:
            new_assumptions = Assumptions(df, curr_idx, self.currency_pair, trade_amount)

        self.recent_tuple = new_assumptions.create_aat_tuple()

    # Adds the AAT correction term (what AAT is trying to learn) to the newest AAT tuple, stores the tuple, then resets
    # the tuple
    def add_correction_term(self, final_trade_amount: float) -> None:
        assert self.recent_tuple is not None

        predicted_amount = self.recent_tuple[-1]
        correction_term = final_trade_amount / predicted_amount

        self.recent_tuple.append(correction_term)
        self.training_data.append(self.recent_tuple)

        if self.auto_aat:
            self.states.append(self.recent_state)
            self.trade_amounts.append(final_trade_amount)

        # Reset the tuple for the next training iteration
        self.recent_tuple = None
        self.recent_state = None

    # Trains a KNN model, saves the model, and saves the AAT correction terms
    def save(self) -> None:
        file_adjustment = '_auto' if self.auto_aat else ''
        name_pair_time_year_str = f'{self.strategy_name}_{self.currency_pair}_{self.time_frame}_{self.year}'

        x = np.array(self.training_data, dtype=float)
        if x.shape[0] == 0:
            print(f'NO TRADES MADE ON {name_pair_time_year_str}')
            return
        x = x[:, 0:-2]
        y = np.array(self.training_data)[:, -1]
        n_neighbors = int(len(x) ** 0.5)

        print(f'X and Y data for {name_pair_time_year_str}')
        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))
        print(f'N neighbors: {n_neighbors}')

        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(x, y)

        correction_terms_file_name = f'../aat/training_data/{name_pair_time_year_str}_aat_correction_terms{file_adjustment}.pickle'
        knn_file_name = f'../aat/training_data/{name_pair_time_year_str}_aat_knn{file_adjustment}.pickle'

        # Save the corrections and KNN model
        with open(correction_terms_file_name, 'wb') as f:
            pickle.dump(y, f)

        with open(knn_file_name, 'wb') as f:
            pickle.dump(knn, f)

        if self.auto_aat:
            assert len(self.trade_amounts) == len(self.states)
            assert len(self.trade_amounts) > 0

            trade_amounts_file = f'../aat/training_data/{name_pair_time_year_str}_trade_amounts.pickle'
            states_file = f'../aat/training_data/{name_pair_time_year_str}_states.pickle'

            with open(trade_amounts_file, 'wb') as f:
                pickle.dump(np.array(self.trade_amounts, dtype=float), f)

            with open(states_file, 'wb') as f:
                pickle.dump(np.array(self.states, dtype=float), f)
