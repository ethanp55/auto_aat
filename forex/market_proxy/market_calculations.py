from datetime import datetime
from forex.market_proxy.trade import TradeType, Trade
import numpy as np


class MarketCalculations(object):
    @staticmethod
    def calculate_day_fees(trade: Trade, currency_pair: str, end_date: datetime) -> float:
        amounts_per_day = [-0.008, -0.01, -0.012] if 'Jpy' in currency_pair else [-0.00008, -0.0001, -0.00012]
        start_date, n_units = trade.start_date, trade.n_units
        curr_fee = np.random.choice(amounts_per_day, p=[0.25, 0.50, 0.25]) * n_units
        num_days = np.busday_count(start_date.date(), end_date.date())

        return num_days * curr_fee

    @staticmethod
    def get_n_units(trade_type: TradeType, stop_loss: float, ask_open: float, bid_open: float, mid_open: float,
                    currency_pair: str, amount_to_risk: float) -> int:
        _, second = currency_pair.split('_')

        pips_to_risk = ask_open - stop_loss if trade_type == TradeType.BUY else stop_loss - bid_open
        pips_to_risk_calc = pips_to_risk * 10000 if second != 'Jpy' else pips_to_risk * 100

        if second == 'Usd':
            per_pip = 0.0001

        else:
            per_pip = 0.0001 / mid_open if second != 'Jpy' else 0.01 / mid_open

        n_units = int(amount_to_risk / (pips_to_risk_calc * per_pip))

        if second == 'Jpy':
            n_units /= 100

        return n_units
