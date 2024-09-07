import glob
import pandas as pd
from pathlib import Path


def compute_total_current(df: pd.DataFrame) -> pd.DataFrame:
    """
     Computes total current from photovoltaic and external supply.

     :param df: DataFrame with 'pv' and 'external_supply' columns.
     :return: Updated DataFrame with a new column 'total_current'.
     """
    df['total_current'] = df['pv'] + df['external_supply']
    return df


def compute_internal_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes internal consumption of energy.

    :param df: DataFrame with 'pv' and 'energy_recovery' columns.
    :return: Updated DataFrame with a new column 'internal_consumption'.
    """
    # calculate internal_consumption
    df['internal_consumption'] = df['pv'] - df['energy_recovery']
    # create a mask for where internal_consumption is less than 0
    mask = df['internal_consumption'] < 0
    # update energy_recovery where mask is True
    df.loc[mask, 'energy_recovery'] = df.loc[mask, 'pv']
    # update internal_consumption where mask is True
    df.loc[mask, 'internal_consumption'] = 0
    return df


def date_to_unix(date: str) -> int:
    """
    Converts date to UNIX timestamp.

    :param date: Date in string format.
    :return: UNIX timestamp.
    """
    datetime = pd.to_datetime(date, dayfirst=True)
    return datetime.value


def remove_broken_counter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to remove rows with broken counters.

    :param df: DataFrame containing power data.
    :return: Filtered DataFrame.
    """
    # for col in df.columns:
        # create a mask where the difference from the previous row is not zero, or the value is zero
        # remove the rows where the value sequence is not changing
    #    mask = (df[col].diff() != 0) | (df[col] == 0)
    #    df = df[mask]

    # Identify where the difference in time from one row to the next is greater than the expected frequency
    # remove the row gap rows
    # seq = 2 * 5
    # mask = df.index.to_series().diff() >= pd.Timedelta(f'{seq}min')
    # df = df.loc[~mask]

    # remove all days in which more than one hour of data points are missing.
    df = df.groupby(pd.Grouper(freq='D')).filter(lambda x: len(x) >= 276)  # ((24*60)/5)-(60/5) = 276

    # Identify the rows between 22:00 and 05:00 and set pv values to 0
    df.loc[df.between_time('22:00', '05:00').index, 'pv'] = 0
    return df


def custom_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom filling of NaN sequences in a DataFrame.

    :param df: DataFrame with possible NaN values.
    :return: DataFrame with filled NaN values.
    """
    # Find the bounds of the NaN sequences
    bounds_start = df.isnull() & ~df.shift(1).isnull()
    bounds_end = df.notnull() & df.shift(1).isnull()
    bounds_end.iloc[0] = False  # Ensure the first index is not an end boundary

    bounds = bounds_start.any(axis=1) | bounds_end.any(axis=1)
    bounds = bounds[bounds].index

    df_list = []

    # Iterate through the bounds and fill each NaN sequence
    for i in range(0, len(bounds), 2):
        # Define start and end of the current NaN sequence
        start = bounds[i]
        end = bounds[i + 1] if i + 1 < len(bounds) else None

        # Get the chunk including the current NaN sequence
        chunk = df.loc[start:end]

        # Calculate the shift as the length of the NaN sequence
        shift = len(chunk)-1
        # shift the dataframe by the length of the chunk
        df_temp = df.fillna(df.shift(shift))
        chunk_filled = df_temp.loc[start:end]
        df_list.append(chunk_filled)

    df_filled = pd.concat(df_list)
    # Add the non-NaN values from the original dataframe
    df_filled = df_filled.combine_first(df)
    return df_filled


def fill_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rounds the DataFrame's index to the nearest 5 minutes and fills missing dates.

    :param df: DataFrame with datetime index.
    :return: DataFrame with rounded and filled datetime index.
    """
    df.index = df.index.floor('T')
    df.index = pd.Series(df.index).dt.round('5min').values
    df = df[~df.index.duplicated(keep='first')]
    dates = pd.date_range(df.index.min(), df.index.max(), freq='5min')
    df = df.reindex(dates)

    df_filled = custom_fill(df)
    return df_filled


def read_csv(directory: str, counter_typ: str, index_col: str) -> pd.DataFrame:
    """
    Reads csv files from a specified directory, concatenates them into a single DataFrame.

    :param directory: Directory containing csv files.
    :param counter_typ: Type of the counter, used in file names.
    :param index_col: Column to be set as the DataFrame's index.
    :return: DataFrame containing all data from csv files.
    """
    root_path = Path(__file__).parent.parent
    df_list = []
    for file in glob.glob(f'{root_path}/{directory}/{counter_typ}/{counter_typ}_tag_*.csv'):
        df_temp = pd.read_csv(file, sep=';').iloc[:, :-1]
        df_temp = df_temp.dropna()
        # convert timestamp to unix time
        df_temp['unix'] = df_temp[index_col].apply(date_to_unix)
        df_temp = df_temp.drop(columns=[index_col])

        # append dataframe to the list
        df_list.append(df_temp)

    # concatenate all dataframes
    df = pd.concat(df_list, ignore_index=True)

    df['unix'] = pd.to_datetime(df['unix'])
    df = df.set_index('unix')
    df = df.sort_index(ascending=True)
    return df
