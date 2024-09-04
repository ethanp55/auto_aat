import pandas as pd
from forex.utils.utils import YEARS


class DataLoader(object):
    @staticmethod
    def load_training_data(currency_pair: str, time_frame: str, year: int, pips_multiplier: int, ) -> pd.DataFrame:
        assert year < YEARS[-1]

        next_year_date = f'{year + 1}-11-01 00:00:00'
        curr_year_date = f'{year}-11-01 00:00:00'

        df = pd.read_csv(f'../data/files/Oanda_{currency_pair}_{time_frame}_2013-2023.csv')
        df.Date = pd.to_datetime(df.Date, utc=True)
        df = df.loc[(df['Date'] >= curr_year_date) & (df['Date'] < next_year_date)]
        df.reset_index(drop=True, inplace=True)

        # Create the labels (what we're trying to predict)
        df['bid_pips_down'] = abs(
            df['Bid_Open'] - df['Bid_Low']) * pips_multiplier
        df['bid_pips_up'] = abs(
            df['Bid_High'] - df['Bid_Open']) * pips_multiplier
        df['ask_pips_down'] = abs(
            df['Ask_Open'] - df['Ask_Low']) * pips_multiplier
        df['ask_pips_up'] = abs(
            df['Ask_High'] - df['Ask_Open']) * pips_multiplier

        return df

    @staticmethod
    def load_simulation_data(currency_pair: str, time_frame: str, optimize: bool, year: int) -> pd.DataFrame:
        if not optimize:
            # We only want to test on data from 2014 and beyond
            assert year > YEARS[0]

        else:
            # We want to stop training after 2022
            assert year < YEARS[-1]

        next_year_date = f'{year + 1}-11-01 00:00:00'
        curr_year_date = f'{year}-11-01 00:00:00'

        df = pd.read_csv(f'../data/files/Oanda_{currency_pair}_{time_frame}_2013-2023.csv')
        df.Date = pd.to_datetime(df.Date, utc=True)
        df = df.loc[(df['Date'] >= curr_year_date) & (df['Date'] < next_year_date)]
        df.reset_index(drop=True, inplace=True)

        return df
