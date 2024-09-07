import copy
import math
import random

import pandas as pd
import numpy as np
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt
from simulation.constant import CHARGING_INTERVAL,TRAIN_SPLIT, VAL_SPLIT, START_DATE, MAX_PV_POWER

sns.set()

PV_IDX, MOD_IDX, DOW_IDX = 1, 2, 3


def plot_supply(df: pd.DataFrame, title: str) -> None:
    """
    Plot the supplied dataframe with two different y-axis.

    :param df: The dataframe containing 'pv' columns
    :param title: The title of the plot
    :return: None
    """
    df['pv_plot'] = df['pv'] * (1 / 1e3)
    fig, ax1 = plt.subplots()

    # Plotting 'pv' on the first y-axis
    ax1.set_xlabel('Time [m]')
    ax1.set_ylabel('PV [kWh]', color='tab:blue', alpha=0.8)
    ax1.plot(df['index'], df['pv_plot'], color='tab:blue', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Turns off grid on the left Axis.
    plt.xlim(0, len(df))
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


class Supply(object):
    def __init__(self,  pv_scaling: float,  pv_noise: bool, seed: int) \
            -> None:
        """
        Initialize the class with the provided parameters.
        :param pv_scaling: The scaling factor for the PV data.
        :param pv_noise:
        :param seed: The random seed
        """
        # data_col, defines the data of the pv power to be considered.
        np.random.seed(seed)
        random.seed(seed)

        self.pv_scaling = pv_scaling
        self.pv_noise = pv_noise

        self.supply = self.__create_supply()
        self.supply_list = []
        # plot_title = f'Season: {season}, PV-Scaling: {pv_scaling}'
        # plot_supply(self.supply, plot_title)

    def __create_supply(self) -> pd.DataFrame:
        """
        Creates a supply dataframe using the provided conditions.

        :return: The supply dataframe.
        """
        df = self.__load_power()
        df = self.__set_day_index(df)

        # add Gaussian noise
        if self.pv_noise:
            df = self.__add_noise(df)

        df['pv'] = df['pv'].round()
        df['index'] = [i for i in range(0, len(df))]
        return df

    def __add_noise(self,df, noise_percentage=0.05) -> pd.DataFrame:
        data = df['pv'].to_numpy()
        time_based_noise_factor = data * noise_percentage
        noise = np.random.normal(0, time_based_noise_factor, len(data))
        noisy_data = data + noise
        noisy_data = np.round(np.clip(noisy_data, 0, MAX_PV_POWER*self.pv_scaling),2)
        df['pv'] = noisy_data
        return df

    def __load_power(self) -> pd.DataFrame:
        """
        Load the power data from a CSV file.

        :return: The power data as a DataFrame.
        """
        df = pd.read_csv('data/counter/pv_supply.csv', sep=';', index_col=0, parse_dates=True)
        df = df[['pv']]
        df = self.__interpolate_data(df)
        df['pv'] = df * self.pv_scaling
        return df[['pv']]


    @staticmethod
    def __local_time_to_utc(df: pd.DataFrame) -> pd.DataFrame:
        # Convert the localized DateTime index to UTC
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous=False, nonexistent='shift_forward')
        df.index = df.index.tz_convert('UTC')
        # Remove duplicates
        duplicates = df.index.duplicated()
        df = df[~duplicates]

        # Define the start and end dates for filtering
        start_date = pd.Timestamp('2022-07-01', tz='UTC')
        end_date = pd.Timestamp('2023-07-01', tz='UTC')

        # Filter the DataFrame to include only dates within the specified range
        df = df.loc[start_date:end_date][:-1]
        return df

    @staticmethod
    def __interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate the dataframe to fill in any missing data.

        :param df: The dataframe to interpolate.
        :return: The interpolated dataframe.
        """
        # add the first and last midnight
        first_index = df.index[0].normalize()
        last_index = (df.index[-1] + pd.DateOffset(days=1)).normalize()
        # Reindex the DataFrame and fill the new values with the previous ones
        df = df.reindex(df.index.union([first_index, last_index]))
        # Use forward fill for the first row
        df.loc[first_index] = df.loc[first_index].fillna(method='bfill')
        # Use backward fill for the last row
        df.loc[last_index] = df.loc[last_index].fillna(method='ffill')

        new_index = pd.date_range(start=first_index, end=last_index, freq=f'{CHARGING_INTERVAL}min')

        # reindex the dataframe to include any missing time steps
        df = df.reindex(new_index)

        # interpolate to fill in missing data
        df = df.interpolate(method='time')
        df = df.iloc[:-1]
        return df

    @staticmethod
    def __smooth_transitions(df: pd.DataFrame, n_smooth_points: int) -> pd.DataFrame:
        """
        Function to smooth the transition of the data in the dataframe.

        :param df: Dataframe that contains the data.
        :param n_smooth_points: Number of data points to be used in the smoothing process.
        :return: Dataframe with smoothed data.
        """
        for i in range(1, len(df)):
            if df.index[i].date() > df.index[i - 1].date():
                start = max(0, i - n_smooth_points)
                end = min(len(df), i + n_smooth_points)
                df.iloc[start:end] = np.linspace(df.iloc[start], df.iloc[end - 1], end - start)
        return df

    def __set_day_index(self, df):
        """

        :param df:
        :return:
        """
        # Create a date range that starts at start_date and has as many periods as the length of df
        start_date = START_DATE
        date_rng = pd.date_range(start=start_date, periods=len(df), freq='T')  # 'T' for minutes
        df = df.set_index(date_rng)

        # Calculate the minute of the day
        df['mod'] = df.index.hour * 60 + df.index.minute
        return df

    def split_supply(self, events: pd.DataFrame, data_set: str = 'train') -> int:
        df = copy.deepcopy(self.supply)

        # Initialize 'park_state' column
        df['park_state'] = 0

        # Set 'park_state' to 1 based on events
        for index, row in events.iterrows():
            if row['departure_time'] + 1 >= len(df) or math.isnan(row['departure_time']):
                df.iloc[row['arrival_time']:len(df) - 1, df.columns.get_loc('park_state')] = 1
            else:
                df.iloc[int(row['arrival_time']):int(row['departure_time'] + 1), df.columns.get_loc('park_state')] = 1

        # Create an auxiliary column that detects the change points
        df['change'] = df['park_state'].diff().fillna(0).abs().astype(int)

        # Label each sequence group
        df['group'] = (df['change'] == 1).cumsum()

        dfs = []
        for _, group in df.groupby('group'):
            # Get the preceding row
            if group.index[0] - pd.Timedelta(minutes=1) in df.index:
                preceding_row = df.loc[[group.index[0] - pd.Timedelta(minutes=1)]]
            else:
                preceding_row = None

            # Get the succeeding row
            if group.index[-1] + pd.Timedelta(minutes=1) in df.index:
                succeeding_row = df.loc[[group.index[-1] + pd.Timedelta(minutes=1)]]
            else:
                succeeding_row = None

            # Concatenate preceding_row, group, and succeeding_row
            segments = [seg for seg in [preceding_row, group, succeeding_row] if seg is not None]
            combined_df = pd.concat(segments, axis=0)

            dfs.append(combined_df)

        dfs = [segment for segment in dfs if segment['park_state'].sum() > 2]

        # Splits the data into train, validation, and test sets based on the specified ratios.
        shuffled_data = dfs.copy()
        random.shuffle(shuffled_data)

        # Calculate the size of each split
        train_size = int(len(shuffled_data) * TRAIN_SPLIT)
        val_size = int(len(shuffled_data) * VAL_SPLIT)
        print(f'Train/val/test: {len(shuffled_data[:train_size])}/{len(shuffled_data[train_size:train_size + val_size])}/{len(shuffled_data[train_size + val_size:])}')
        if data_set == 'all':
            self.supply_list = dfs
        elif data_set == 'train':
            self.supply_list = shuffled_data[:train_size]
        elif data_set == 'tune':
            self.supply_list = shuffled_data[train_size:]
        elif data_set == 'val':
            self.supply_list = shuffled_data[train_size:train_size + val_size]
        else:
            self.supply_list = shuffled_data[train_size + val_size:]

        self.supply = self.supply[['index', 'pv', 'mod']].values
        self.supply_list = [df[['index', 'pv', 'mod']].values for df in self.supply_list]
        return len(self.supply_list)

    def get_supply(self, time_step: int, env_id: int) -> Tuple[float, int]:
        """
        Get the pv power at a particular time step.

        :param time_step: The time step to consider.
        :param env_id:
        :return: A tuple containing the power, minute of the day and day of the week.
        """
        if env_id == -1:
            row = self.supply[time_step]
        else:
            env = self.supply_list[env_id]
            row = env[time_step]

        return row[PV_IDX], row[MOD_IDX]
