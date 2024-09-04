import numpy as np
import pandas as pd


class TechnicalIndicators(object):
    @staticmethod
    def adx(high, low, close, lookback=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha=1 / lookback).mean()

        return adx_smooth

    @staticmethod
    def stoch(high, low, close, lookback=14):
        high_lookback = high.rolling(lookback).max()
        low_lookback = low.rolling(lookback).min()
        slow_k = (close - low_lookback) * 100 / (high_lookback - low_lookback)
        slow_d = slow_k.rolling(3).mean()

        return slow_k, slow_d

    @staticmethod
    def stoch_rsi(rsi, k_window=3, d_window=3, window=14):
        min_val = rsi.rolling(window=window, center=False).min()
        max_val = rsi.rolling(window=window, center=False).max()

        stoch = ((rsi - min_val) / (max_val - min_val)) * 100

        slow_k = stoch.rolling(window=k_window, center=False).mean()

        slow_d = slow_k.rolling(window=d_window, center=False).mean()

        return slow_k, slow_d

    @staticmethod
    def chop(df, lookback=14):
        atr1 = TechnicalIndicators.atr(df['Mid_High'], df['Mid_Low'], df['Mid_Close'], lookback=1)
        high, low = df['Mid_High'], df['Mid_Low']

        chop = np.log10(
            atr1.rolling(lookback).sum() / (high.rolling(lookback).max() - low.rolling(lookback).min())) / np.log10(
            lookback)

        return chop

    @staticmethod
    def vo(volume, short_lookback=5, long_lookback=10):
        short_ema = pd.Series.ewm(volume, span=short_lookback).mean()
        long_ema = pd.Series.ewm(volume, span=long_lookback).mean()

        volume_oscillator = (short_ema - long_ema) / long_ema

        return volume_oscillator

    @staticmethod
    def williams_r(highs, lows, closes, length=21, ema_length=15):
        highest_highs = highs.rolling(window=length).max()
        lowest_lows = lows.rolling(window=length).min()

        willy = 100 * (closes - highest_highs) / (highest_highs - lowest_lows)
        willy_ema = pd.Series.ewm(willy, span=ema_length).mean()

        return willy, willy_ema

    @staticmethod
    def atr(high, low, close, lookback=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        return true_range.rolling(lookback).mean()

    @staticmethod
    def atr_bands(high, low, close, lookback=14, atr_multiplier=3):
        scaled_atr_vals = TechnicalIndicators.atr(high, low, close, lookback) * atr_multiplier
        lower_band = close - scaled_atr_vals
        upper_band = close + scaled_atr_vals

        return lower_band, upper_band

    @staticmethod
    def rsi(closes, periods=14):
        close_delta = closes.diff()

        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))

        return rsi

    @staticmethod
    def qqe_mod(closes, rsi_period=6, smoothing=5, qqe_factor=3, threshold=3, mult=0.35, sma_length=50):
        Rsi = TechnicalIndicators.rsi(closes, rsi_period)
        RsiMa = Rsi.ewm(span=smoothing).mean()
        AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
        Wilders_Period = rsi_period * 2 - 1
        MaAtrRsi = AtrRsi.ewm(span=Wilders_Period).mean()
        dar = MaAtrRsi.ewm(span=Wilders_Period).mean() * qqe_factor

        longband = pd.Series(0.0, index=Rsi.index)
        shortband = pd.Series(0.0, index=Rsi.index)
        trend = pd.Series(0, index=Rsi.index)

        DeltaFastAtrRsi = dar
        RSIndex = RsiMa
        newshortband = RSIndex + DeltaFastAtrRsi
        newlongband = RSIndex - DeltaFastAtrRsi
        longband = pd.Series(np.where((RSIndex.shift(1) > longband.shift(1)) & (RSIndex > longband.shift(1)),
                                      np.maximum(longband.shift(1), newlongband), newlongband))
        shortband = pd.Series(np.where((RSIndex.shift(1) < shortband.shift(1)) & (RSIndex < shortband.shift(1)),
                                       np.minimum(shortband.shift(1), newshortband), newshortband))
        cross_1 = (longband.shift(1) < RSIndex) & (longband > RSIndex)
        cross_2 = (RSIndex > shortband.shift(1)) & (RSIndex.shift(1) < shortband)
        trend = np.where(cross_2, 1, np.where(cross_1, -1, trend.shift(1).fillna(1)))
        FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband))

        basis = (FastAtrRsiTL - 50).rolling(sma_length).mean()
        dev = mult * (FastAtrRsiTL - 50).rolling(sma_length).std()
        upper = basis + dev
        lower = basis - dev

        Greenbar1 = RsiMa - 50 > threshold
        Greenbar2 = RsiMa - 50 > upper

        Redbar1 = RsiMa - 50 < 0 - threshold
        Redbar2 = RsiMa - 50 < lower

        Greenbar = Greenbar1 & Greenbar2
        Redbar = Redbar1 & Redbar2

        return Greenbar, Redbar, RsiMa - 50

    @staticmethod
    def supertrend(barsdata, atr_len=10, mult=3):
        curr_atr = TechnicalIndicators.atr(barsdata['Mid_High'], barsdata['Mid_Low'], barsdata['Mid_Close'],
                                           lookback=atr_len)
        highs, lows = barsdata['Mid_High'], barsdata['Mid_Low']
        hl2 = (highs + lows) / 2
        final_upperband = hl2 + mult * curr_atr
        final_lowerband = hl2 - mult * curr_atr

        # initialize Supertrend column to True
        supertrend = [True] * len(barsdata)

        close = barsdata['Mid_Close']

        for i in range(1, len(barsdata.index)):
            curr, prev = i, i - 1

            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True

            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False

            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]

                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

        return supertrend, final_upperband, final_lowerband

    @staticmethod
    def bollinger_bands(barsdata, length=20, mult=2.0):
        m_avg = barsdata['Mid_Close'].rolling(window=length).mean()
        m_std = barsdata['Mid_Close'].rolling(window=length).std(ddof=0)
        lower_bb = m_avg - mult * m_std
        upper_bb = m_avg + mult * m_std

        return lower_bb, upper_bb

    @staticmethod
    def keltner_channels(barsdata, length=20, mult=2.0):
        tr0 = abs(barsdata['Mid_High'] - barsdata['Mid_Low'])
        tr1 = abs(barsdata['Mid_High'] - barsdata['Mid_Close'].shift())
        tr2 = abs(barsdata['Mid_Low'] - barsdata['Mid_Close'].shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        range_ma = tr.rolling(window=length).mean()
        m_avg = barsdata['Mid_Close'].rolling(window=length).mean()
        upper_kc = m_avg + range_ma * mult
        lower_kc = m_avg - range_ma * mult

        return lower_kc, upper_kc

    @staticmethod
    def squeeze(barsdata, length=20, length_kc=20, mult=1.5):
        # Bollinger bands
        m_avg = barsdata['Mid_Close'].rolling(window=length).mean()
        m_std = barsdata['Mid_Close'].rolling(window=length).std(ddof=0)
        upper_bb = m_avg + mult * m_std
        lower_bb = m_avg - mult * m_std

        # Keltner channel
        tr0 = abs(barsdata['Mid_High'] - barsdata['Mid_Low'])
        tr1 = abs(barsdata['Mid_High'] - barsdata['Mid_Close'].shift())
        tr2 = abs(barsdata['Mid_Low'] - barsdata['Mid_Close'].shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        range_ma = tr.rolling(window=length_kc).mean()
        upper_kc = m_avg + range_ma * mult
        lower_kc = m_avg - range_ma * mult

        # Squeeze
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

        return squeeze_on

    @staticmethod
    def squeeze_pro(barsdata, length=20, length_kc=20, bb_mult=2.0, kc_mult_high=1.0, kc_mult_mid=1.5, kc_mult_low=2.0):
        # Bollinger bands
        m_avg = barsdata['Mid_Close'].rolling(window=length).mean()
        m_std = barsdata['Mid_Close'].rolling(window=length).std(ddof=0)
        bb_upper = m_avg + bb_mult * m_std
        bb_lower = m_avg - bb_mult * m_std

        # Keltner channel
        tr0 = abs(barsdata['Mid_High'] - barsdata['Mid_Low'])
        tr1 = abs(barsdata['Mid_High'] - barsdata['Mid_Close'].shift())
        tr2 = abs(barsdata['Mid_Low'] - barsdata['Mid_Close'].shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        range_ma = tr.rolling(window=length_kc).mean()
        kc_upper_high = m_avg + range_ma * kc_mult_high
        kc_lower_high = m_avg - range_ma * kc_mult_high
        kc_upper_mid = m_avg + range_ma * kc_mult_mid
        kc_lower_mid = m_avg - range_ma * kc_mult_mid
        kc_upper_low = m_avg + range_ma * kc_mult_low
        kc_lower_low = m_avg - range_ma * kc_mult_low

        # Squeeze
        low_squeeze = (bb_lower >= kc_lower_low) | (bb_upper <= kc_upper_low)  # Black
        mid_squeeze = (bb_lower >= kc_lower_mid) | (bb_upper <= kc_upper_mid)  # Yellow
        high_squeeze = (bb_lower >= kc_lower_high) | (bb_upper <= kc_upper_high)  # Red

        squeeze_values = np.where(high_squeeze, 'red',
                                  np.where(mid_squeeze, 'yellow', np.where(low_squeeze, 'black', 'green')))

        # Momentum
        highest_high = barsdata['Mid_High'].rolling(window=length).max()
        lowest_low = barsdata['Mid_Low'].rolling(window=length).min()
        avg_high_low = (highest_high + lowest_low) / 2
        avg_avg_high_low_sma = (avg_high_low + m_avg) / 2
        diff = barsdata['Mid_Close'] - avg_avg_high_low_sma
        squeeze_momentum = diff.rolling(window=length).apply(
            lambda x: np.polyfit(np.arange(length), x, 1)[0] * (length - 1) + np.polyfit(np.arange(length), x, 1)[1],
            raw=True)
        iff_1 = np.where(squeeze_momentum > squeeze_momentum.shift(), 'aqua', 'blue')
        iff_2 = np.where(squeeze_momentum < squeeze_momentum.shift(), 'red', 'yellow')
        squeeze_momentum_color = np.where(squeeze_momentum > 0, iff_1, iff_2)

        return squeeze_values, squeeze_momentum_color

    @staticmethod
    def n_macd(df, short_len=12, long_len=21, signal_period=9, lookback=50):
        sh, lon = pd.Series.ewm(df['Mid_Close'], span=short_len).mean(), pd.Series.ewm(df['Mid_Close'],
                                                                                       span=long_len).mean()
        ratio = np.minimum(sh, lon) / np.maximum(sh, lon)
        macd = pd.Series(np.where(sh > lon, 2 - ratio, ratio) - 1)
        n_macd = (macd - macd.rolling(lookback).min()) / (
                macd.rolling(lookback).max() - macd.rolling(lookback).min() + 0.000001) * 2 - 1

        weights = np.arange(1, signal_period + 1)
        n_macdsignal = n_macd.rolling(window=signal_period).apply(lambda x: np.dot(x, weights) / weights.sum(),
                                                                  raw=True)

        return n_macd, n_macdsignal

    @staticmethod
    def smma(closes, length):
        smma = []

        for i in range(len(closes)):
            if i < length:
                smma.append(closes.iloc[:i + 1, ].rolling(length).mean().iloc[-1,])

            else:
                smma.append((smma[i - 1] * (length - 1) + closes[i]) / length)

        return pd.Series(smma)

    @staticmethod
    def impulse_macd(high, low, close, ma_len=34, signal_period=9):
        def _calc_zlema(series, length):
            ema1 = pd.Series.ewm(series, span=length).mean()
            ema2 = pd.Series.ewm(ema1, span=length).mean()
            diff = ema1 - ema2

            return ema1 + diff

        hlc3 = (high + low + close) / 3
        hi = TechnicalIndicators.smma(high, ma_len)
        lo = TechnicalIndicators.smma(low, ma_len)
        mi = _calc_zlema(hlc3, ma_len)
        md = np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0))
        sb = pd.Series(md).rolling(signal_period).mean()

        return md, sb

    @staticmethod
    def psar(barsdata, iaf=0.02, maxaf=0.2):
        length = len(barsdata)
        high = list(barsdata['Mid_High'])
        low = list(barsdata['Mid_Low'])
        close = list(barsdata['Mid_Close'])
        psar = close[0:len(close)]
        bull = True
        af = iaf
        hp = high[0]
        lp = low[0]
        for i in range(2, length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            reverse = False
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + iaf, maxaf)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + iaf, maxaf)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]
        return psar

    @staticmethod
    def beep_boop(row):
        macdhist, ema50, mid_low, mid_high = row[['macdhist', 'ema50', 'Mid_Low', 'Mid_High']]

        if float(macdhist) > 0 and float(mid_low) > float(ema50):
            return 1

        elif float(macdhist) < 0 and float(mid_high) < float(ema50):
            return 2

        else:
            return 0

    @staticmethod
    def fractal(lows, highs, window=20):
        assert len(lows) == len(highs)

        fractal_period = 2 * window + 1

        is_support = lows.rolling(fractal_period, center=True).apply(
            lambda x: x[window] == min(x), raw=True)
        is_resistance = highs.rolling(fractal_period, center=True).apply(
            lambda x: x[window] == max(x), raw=True)

        is_support_indices = pd.Series(is_support.index[is_support == 1.0])
        is_resistance_indices = pd.Series(
            is_resistance.index[is_resistance == 1.0])

        support_fractal_vals = lows[is_support_indices].reindex(lows.index).ffill()
        resistance_fractal_vals = highs[is_resistance_indices].reindex(
            highs.index).ffill()

        return support_fractal_vals, resistance_fractal_vals

    @staticmethod
    def choc(closes, support_fractals, resistance_fractals):
        broke_down = closes < support_fractals
        broke_up = closes > resistance_fractals

        assert len(broke_down) == len(broke_up)

        choc, prev_val = [], None

        for i in range(len(broke_down)):
            # First occurrence
            if prev_val is None:
                if broke_down[i]:
                    prev_val = 'broke_down'
                    choc.append('broke_down')

                elif broke_up[i]:
                    prev_val = 'broke_up'
                    choc.append('broke_up')

                else:
                    choc.append('na')

            # All other occurences
            elif broke_down[i] and prev_val == 'broke_up':
                prev_val = 'broke_down'
                choc.append('broke_down')

            elif broke_up[i] and prev_val == 'broke_down':
                prev_val = 'broke_up'
                choc.append('broke_up')

            else:
                choc.append('na')

        return pd.Series(choc)

    @staticmethod
    def format_data_for_ml_model(df: pd.DataFrame) -> pd.DataFrame:
        formatted_df = df.copy()
        formatted_df['rsi'] = TechnicalIndicators.rsi(formatted_df['Mid_Close'])
        formatted_df['rsi_sma'] = formatted_df['rsi'].rolling(50).mean()
        formatted_df['adx'] = TechnicalIndicators.adx(formatted_df['Mid_High'], formatted_df['Mid_Low'],
                                                      formatted_df['Mid_Close'])
        formatted_df['chop'] = TechnicalIndicators.chop(formatted_df)
        formatted_df['vo'] = TechnicalIndicators.vo(formatted_df['Volume'])
        formatted_df['qqe_up'], formatted_df['qqe_down'], formatted_df['qqe_val'] = \
            TechnicalIndicators.qqe_mod(formatted_df['Mid_Close'])
        formatted_df['rsi_up'] = formatted_df['rsi'] > formatted_df['rsi_sma']
        formatted_df['adx_large'] = formatted_df['adx'] > 30
        formatted_df['chop_small'] = formatted_df['chop'] < 0.5
        formatted_df['vo_positive'] = formatted_df['vo'] > 0
        formatted_df['squeeze_on'] = TechnicalIndicators.squeeze(formatted_df)
        formatted_df['macd'] = pd.Series.ewm(formatted_df['Mid_Close'], span=12).mean() - \
                               pd.Series.ewm(formatted_df['Mid_Close'], span=26).mean()
        formatted_df['macdsignal'] = pd.Series.ewm(formatted_df['macd'], span=9).mean()

        formatted_df.dropna(inplace=True)
        formatted_df.reset_index(drop=True, inplace=True)

        formatted_df.drop(['Date', 'Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High', 'Ask_Low',
                           'Ask_Close', 'Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close', 'Volume'], axis=1, inplace=True)

        return formatted_df

    @staticmethod
    def format_for_all_possible_strategies(df: pd.DataFrame) -> pd.DataFrame:
        formatted_df = df.copy()
        formatted_df['atr'] = TechnicalIndicators.atr(formatted_df['Mid_High'], formatted_df['Mid_Low'],
                                                      formatted_df['Mid_Close'])
        formatted_df['lower_atr_band'], formatted_df['upper_atr_band'] = TechnicalIndicators.atr_bands(
            formatted_df['Mid_High'], formatted_df['Mid_Low'], formatted_df['Mid_Close'])
        formatted_df['ema200'] = pd.Series.ewm(formatted_df['Mid_Close'], span=200).mean()
        formatted_df['ema100'] = pd.Series.ewm(formatted_df['Mid_Close'], span=100).mean()
        formatted_df['ema50'] = pd.Series.ewm(formatted_df['Mid_Close'], span=50).mean()
        formatted_df['smma200'] = TechnicalIndicators.smma(formatted_df['Mid_Close'], 200)
        formatted_df['smma100'] = TechnicalIndicators.smma(formatted_df['Mid_Close'], 100)
        formatted_df['smma50'] = TechnicalIndicators.smma(formatted_df['Mid_Close'], 50)
        formatted_df['bid_pips_down'] = abs(formatted_df['Bid_Open'] - formatted_df['Bid_Low'])
        formatted_df['bid_pips_up'] = abs(formatted_df['Bid_High'] - formatted_df['Bid_Open'])
        formatted_df['ask_pips_down'] = abs(formatted_df['Ask_Open'] - formatted_df['Ask_Low'])
        formatted_df['ask_pips_up'] = abs(formatted_df['Ask_High'] - formatted_df['Ask_Open'])
        formatted_df['rsi'] = TechnicalIndicators.rsi(formatted_df['Mid_Close'])
        formatted_df['rsi_sma'] = formatted_df['rsi'].rolling(50).mean()
        formatted_df['adx'] = TechnicalIndicators.adx(formatted_df['Mid_High'], formatted_df['Mid_Low'],
                                                      formatted_df['Mid_Close'])
        formatted_df['chop'] = TechnicalIndicators.chop(formatted_df)
        formatted_df['vo'] = TechnicalIndicators.vo(formatted_df['Volume'])
        formatted_df['rsi_up'] = formatted_df['rsi'] > formatted_df['rsi_sma']
        formatted_df['adx_large'] = formatted_df['adx'] > 30
        formatted_df['chop_small'] = formatted_df['chop'] < 0.5
        formatted_df['vo_positive'] = formatted_df['vo'] > 0
        formatted_df['squeeze_on'] = TechnicalIndicators.squeeze(formatted_df)
        formatted_df['macd'] = pd.Series.ewm(formatted_df['Mid_Close'], span=12).mean() - \
                               pd.Series.ewm(formatted_df['Mid_Close'], span=26).mean()
        formatted_df['macdsignal'] = pd.Series.ewm(formatted_df['macd'], span=9).mean()
        formatted_df['macdhist'] = formatted_df['macd'] - formatted_df['macdsignal']
        formatted_df['beep_boop'] = formatted_df.apply(TechnicalIndicators.beep_boop, axis=1)
        fractal_window = 50
        formatted_df['support_fractal'], formatted_df['resistance_fractal'] = TechnicalIndicators.fractal(
            formatted_df['Mid_Low'], formatted_df['Mid_High'], fractal_window)
        formatted_df['support_fractal'], formatted_df['resistance_fractal'] = formatted_df['support_fractal'].shift(
            fractal_window), formatted_df['resistance_fractal'].shift(fractal_window)
        formatted_df['choc'] = TechnicalIndicators.choc(formatted_df['Mid_Close'], formatted_df['support_fractal'],
                                                        formatted_df['resistance_fractal'])
        formatted_df['sar'] = TechnicalIndicators.psar(formatted_df)
        formatted_df['lower_kc'], formatted_df['upper_kc'] = TechnicalIndicators.keltner_channels(formatted_df)
        formatted_df['lower_bb'], formatted_df['upper_bb'] = TechnicalIndicators.bollinger_bands(formatted_df)
        formatted_df['qqe_up'], formatted_df['qqe_down'], formatted_df['qqe_val'] = \
            TechnicalIndicators.qqe_mod(formatted_df['Mid_Close'])
        formatted_df['supertrend'], formatted_df['supertrend_ub'], formatted_df[
            'supertrend_lb'] = TechnicalIndicators.supertrend(formatted_df)
        formatted_df['slowk'], formatted_df['slowd'] = TechnicalIndicators.stoch(formatted_df['Mid_High'],
                                                                                 formatted_df['Mid_Low'],
                                                                                 formatted_df['Mid_Close'])
        formatted_df['slowk_rsi'], formatted_df['slowd_rsi'] = TechnicalIndicators.stoch_rsi(formatted_df['rsi'])
        formatted_df['squeeze_value'], formatted_df['squeeze_momentum_color'] = TechnicalIndicators.squeeze_pro(
            formatted_df)
        formatted_df['n_macd'], formatted_df['n_macdsignal'] = TechnicalIndicators.n_macd(formatted_df)
        formatted_df['impulse_macd'], formatted_df['impulse_macdsignal'] = TechnicalIndicators.impulse_macd(
            formatted_df['Mid_High'], formatted_df['Mid_Low'], formatted_df['Mid_Close'])

        formatted_df.dropna(inplace=True)
        formatted_df.reset_index(drop=True, inplace=True)

        return formatted_df
