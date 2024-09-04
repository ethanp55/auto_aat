from dataclasses import dataclass
from datetime import datetime
import enum
from typing import Optional


class TradeType(enum.Enum):
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    trade_type: TradeType
    open_price: float
    stop_loss: float
    stop_gain: Optional[float]
    n_units: int
    pips_risked: float
    start_date: datetime
    currency_pair: str

    def __post_init__(self):
        rounding = 3 if 'Jpy' in self.currency_pair else 5
        self.open_price = round(self.open_price, rounding)
        self.stop_loss = round(self.stop_loss, rounding)
        self.stop_gain = round(self.stop_gain, rounding) if self.stop_gain is not None else None
