from forex.runner.simulation_runner import SimulationRunner
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
from forex.strategies.supertrend import Supertrend
from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def train_aat() -> None:
    strategy_types = [BarMovement, BeepBoop, BollingerBands, Choc, KeltnerChannels, MACrossover, MACD, MACDKeyLevel,
                      MACDStochastic, PSAR, RSI, SqueezePro, Stochastic, Supertrend]
    auto_aat_vals = [True]

    for auto_aat in auto_aat_vals:
        for currency_pair in CURRENCY_PAIRS:
            for time_frame in TIME_FRAMES:
                for year in YEARS[:-1]:
                    strategies_to_train = [strategy_type() for strategy_type in strategy_types]

                    for strategy in strategies_to_train:
                        SimulationRunner.run_simulation(strategy=strategy, currency_pair=currency_pair,
                                                        time_frame=time_frame, year=year, optimize=True,
                                                        train_aat=True, auto_aat_tuned=True)


if __name__ == "__main__":
    train_aat()
