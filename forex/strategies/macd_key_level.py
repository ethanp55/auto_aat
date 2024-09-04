from forex.market_proxy.market_calculations import MarketCalculations
from forex.market_proxy.market_simulation_results import MarketSimulationResults
from forex.market_proxy.trade import Trade, TradeType
from pandas import DataFrame
from forex.strategies.strategy import Strategy
from typing import Optional


class MACDKeyLevel(Strategy):
    def __init__(self, starting_idx: int = 2,
                 percent_to_risk: float = 0.02, macd_type: str = 'macd', ma_key: Optional[str] = 'smma200',
                 macd_threshold: float = 0.0, invert: bool = False, use_tsl: bool = False,
                 pips_to_risk: Optional[int] = 50,
                 pips_to_risk_atr_multiplier: float = 2.0, risk_reward_ratio: Optional[float] = 1.5,
                 close_to_level_atr_multiplier: float = 1.0, close_trade_incrementally: bool = False) -> None:
        super().__init__(starting_idx, percent_to_risk, 'MACDKeyLevel')
        self.macd_type, self.ma_key, self.macd_threshold, self.invert, self.use_tsl, self.pips_to_risk, self.pips_to_risk_atr_multiplier, \
        self.risk_reward_ratio, self.close_to_level_atr_multiplier, self.close_trade_incrementally = macd_type, ma_key, macd_threshold, invert, use_tsl, pips_to_risk, \
                                                                                                     pips_to_risk_atr_multiplier, risk_reward_ratio, close_to_level_atr_multiplier, close_trade_incrementally

    def place_trade(self, curr_idx: int, strategy_data: DataFrame, currency_pair: str, account_balance: float) -> \
            Optional[Trade]:
        # Determine which type of macd to use
        macd_key, macdsignal_key = ('n_macd', 'n_macdsignal') if self.macd_type == 'n_macd' else (
            ('impulse_macd', 'impulse_macdsignal') if self.macd_type == 'impulse_macd' else ('macd', 'macdsignal'))

        # Grab the needed values from the market data
        macd2, macdsignal2 = strategy_data.loc[strategy_data.index[curr_idx - 2], [macd_key, macdsignal_key]]
        macd1, macdsignal1, atr, mid_close = strategy_data.loc[
            strategy_data.index[curr_idx - 1], [macd_key, macdsignal_key, 'atr', 'Mid_Close']]
        ma = strategy_data.loc[strategy_data.index[curr_idx - 1], self.ma_key] if self.ma_key is not None else None
        pips_level_rounding = 0 if 'Jpy' in currency_pair else 2

        # Determine if there is a buy or sell signal
        crossed_up = macd2 < macdsignal2 and macd1 > macdsignal1 and max([macd2, macdsignal2, macd1, macdsignal1]) < 0
        crossed_down = macd2 > macdsignal2 and macd1 < macdsignal1 and min([macd2, macdsignal2, macd1, macdsignal1]) > 0
        macd_final_threshold = self.macd_threshold * atr if self.macd_type != 'n_macd' else self.macd_threshold
        macd_large_enough = min([abs(macd1), abs(macdsignal1), abs(macd2), abs(macdsignal2)]) >= macd_final_threshold
        buy_ma_condition_met = mid_close > ma if ma is not None else True
        sell_ma_condition_met = mid_close < ma if ma is not None else True
        nearest_level = round(mid_close, pips_level_rounding)
        close_to_nearest_level = abs(mid_close - nearest_level) < atr * self.close_to_level_atr_multiplier

        buy_signal = crossed_up and macd_large_enough and buy_ma_condition_met and close_to_nearest_level
        sell_signal = crossed_down and macd_large_enough and sell_ma_condition_met and close_to_nearest_level

        if self.invert:
            buy_signal, sell_signal = sell_signal, buy_signal

        # If there is a signal, place a trade (assuming the spread is small enough)
        if buy_signal or sell_signal:
            curr_date, curr_ao, curr_bo, curr_mo, curr_bh, curr_al = strategy_data.loc[
                strategy_data.index[curr_idx], ['Date', 'Ask_Open', 'Bid_Open', 'Mid_Open', 'Bid_High', 'Ask_Low']]
            spread = abs(curr_ao - curr_bo)
            divider = 100 if 'Jpy' in currency_pair else 10000
            pips_to_risk = self.pips_to_risk / divider if self.pips_to_risk is not None else None
            sl_pips = pips_to_risk if pips_to_risk is not None else atr * self.pips_to_risk_atr_multiplier

            if buy_signal:
                open_price = curr_ao
                stop_loss = open_price - sl_pips

                if stop_loss < open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.BUY
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    stop_gain = None if self.risk_reward_ratio is None else open_price + (
                            sl_pips * self.risk_reward_ratio)

                    return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                                 currency_pair)

            elif sell_signal:
                open_price = curr_bo
                stop_loss = open_price + sl_pips

                if stop_loss > open_price and spread <= sl_pips * 0.1:
                    trade_type = TradeType.SELL
                    amount_to_risk = account_balance * self.percent_to_risk
                    n_units = MarketCalculations.get_n_units(trade_type, stop_loss, curr_ao, curr_bo, curr_mo,
                                                             currency_pair, amount_to_risk)
                    stop_gain = None if self.risk_reward_ratio is None else open_price - (
                            sl_pips * self.risk_reward_ratio)

                    return Trade(trade_type, open_price, stop_loss, stop_gain, n_units, sl_pips, curr_date,
                                 currency_pair)

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
