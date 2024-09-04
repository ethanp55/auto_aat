from pandas import DataFrame
from typing import List


class Assumptions:
    def __init__(self, df: DataFrame, curr_idx: int, currency_pair: str, trade_amount: float,
                 calculate: bool = True) -> None:
        if calculate:
            # Create assumptions from the current row of data
            mid_close, atr = df.loc[df.index[curr_idx - 1], ['Mid_Close', 'atr']]

            ema200, ema100, ema50 = df.loc[df.index[curr_idx - 1], ['ema200', 'ema100', 'ema50']]
            self.up_trend_long, self.up_trend_mid, self.up_trend_short = \
                mid_close > ema200, mid_close > ema100, mid_close > ema50

            smma200, smma100, smma50 = df.loc[df.index[curr_idx - 1], ['smma200', 'smma100', 'smma50']]
            self.s_up_trend_long, self.s_up_trend_mid, self.s_up_trend_short = \
                mid_close > smma200, mid_close > smma100, mid_close > smma50

            lower_atr_band, upper_atr_band = df.loc[df.index[curr_idx - 1], ['lower_atr_band', 'upper_atr_band']]
            self.sideways_long, self.sideways_mid, self.sideways_short = lower_atr_band < ema200 < upper_atr_band, \
                                                                         lower_atr_band < ema100 < upper_atr_band, \
                                                                         lower_atr_band < ema50 < upper_atr_band
            self.s_sideways_long, self.s_sideways_mid, self.s_sideways_short = lower_atr_band < smma200 < upper_atr_band, \
                                                                               lower_atr_band < smma100 < upper_atr_band, \
                                                                               lower_atr_band < smma50 < upper_atr_band

            rsi, rsi_sma = df.loc[df.index[curr_idx - 1], ['rsi', 'rsi_sma']]
            self.rsi_up, self.rsi_overbought, self.rsi_oversold = rsi > rsi_sma, rsi > 80, rsi < 20

            adx, chop, vo = df.loc[df.index[curr_idx - 1], ['adx', 'chop', 'vo']]
            self.adx_massive, self.adx_large, self.adx_mid = adx > 40, adx > 30, adx > 20
            self.chop_small, self.chop_mid = chop < 0.4, chop < 0.5
            self.vo_positive = vo > 0

            self.qqe_up, self.qqe_down, self.squeeze_on = \
                df.loc[df.index[curr_idx - 1], ['qqe_up', 'qqe_down', 'squeeze_on']]

            support_fractal, resistance_fractal = df.loc[df.index[curr_idx - 1], ['support_fractal', 'resistance_fractal']]
            self.close_to_support, self.close_to_resistance = \
                abs(mid_close - support_fractal) < atr, abs(mid_close - resistance_fractal) < atr
            self.above_support, self.below_resistance = mid_close > support_fractal, mid_close < resistance_fractal

            squeeze_momentum_color = df.loc[df.index[curr_idx - 1], 'squeeze_momentum_color']
            self.squeeze_momentum_bullish = squeeze_momentum_color in {'aqua', 'blue'}

            lower_kc, upper_kc, lower_bb, upper_bb = \
                df.loc[df.index[curr_idx - 1], ['lower_kc', 'upper_kc', 'lower_bb', 'upper_bb']]
            self.lower_kc_broken, self.upper_kc_broken = mid_close < lower_kc, mid_close > upper_kc
            self.lower_bb_broken, self.upper_bb_broken = mid_close < lower_bb, mid_close > upper_bb

            pips_level_rounding = 0 if 'Jpy' in currency_pair else 2
            nearest_level = round(mid_close, pips_level_rounding)
            self.close_to_nearest_level = abs(mid_close - nearest_level) < atr

            # Store the prediction (how much the trade is worth)
            self.prediction = trade_amount

    def set_vals(self, *args) -> None:
        self.up_trend_long, self.up_trend_mid, self.up_trend_short, self.s_up_trend_long, self.s_up_trend_mid, \
        self.s_up_trend_short, self.sideways_long, self.sideways_mid, self.sideways_short, self.s_sideways_long, \
        self.s_sideways_mid, self.s_sideways_short, self.rsi_up, self.rsi_overbought, self.rsi_oversold, \
        self.adx_massive, self.adx_large, self.adx_mid, self.chop_small, self.chop_mid, self.vo_positive, \
        self.qqe_up, self.qqe_down, self.squeeze_on, self.close_to_support, self.close_to_resistance, \
        self.above_support, self.below_resistance, self.squeeze_momentum_bullish, self.lower_kc_broken, \
        self.upper_kc_broken, self.lower_bb_broken, self.upper_bb_broken, self.close_to_nearest_level, \
        self.prediction = args

    def create_aat_tuple(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [self.__getattribute__(name) for name in attribute_names]

        return tup
