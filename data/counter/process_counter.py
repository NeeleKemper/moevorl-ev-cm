import os
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from utils import read_csv, compute_total_current, compute_internal_consumption, fill_dates, \
    remove_broken_counter
from simulation.constant import MAX_PV_POWER

warnings.filterwarnings('ignore')
sns.set()


# sns.set_style("whitegrid")


def remove_chp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the Combined Heat and Power (CHP) column from the dataframe.

    :param df: The dataframe with power data.
    :return: The dataframe with the CHP column removed.
    """
    # compute the difference
    diff = df['energy_recovery'] - df['chp']

    # create a mask for the rows where diff is less than zero
    mask = diff < 0

    # update 'external_supply' and 'energy_recovery' for the rows where diff is less than zero
    df.loc[mask, 'external_supply'] = df.loc[mask, 'external_supply'] + abs(diff[mask])
    df.loc[mask, 'energy_recovery'] = 0

    # update 'energy_recovery' for the rows where diff is not less than zero
    df.loc[~mask, 'energy_recovery'] = diff[~mask]
    df = df.drop(columns=['chp'])
    return df


def process_zl() -> pd.DataFrame:
    """
    Reads and processes the CSV file containing power data.

    :return: A dataframe containing processed power data.
    """
    df = read_csv('counter', 'zl', 'tkt=300,Leistung_in_W')
    df = df.rename(
        columns={'EL:GesamtStrom': 'total_current', 'EL:Stromlieferung PV': 'pv',
                 'EL:Strombezug Trafo': 'external_supply', 'EL:Stromlieferung Trafo': 'energy_recovery',
                 'EL:Eigenverbrauch': 'internal_consumption', 'EL:Stromlieferung BHKW': 'chp'})

    others = ['EL:AC Leistung', 'EL:DC Leistung', 'internal_consumption', 'total_current']
    df = df.drop(columns=others)

    df = remove_broken_counter(df)
    df = remove_chp(df)
    df = fill_dates(df)
    df = compute_total_current(df)
    df = compute_internal_consumption(df)
    return df


def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
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

    new_index = pd.date_range(start=first_index, end=last_index, freq=f'5T')

    # reindex the dataframe to include any missing time steps
    df = df.reindex(new_index)

    # interpolate to fill in missing data
    df = df.interpolate(method='time')
    df = df.iloc[:-1]
    return df


def custom_month_year_formatter(x):
    """Custom formatter to show month and year for January and only month for other months."""
    date = mdates.num2date(x)
    if date.month == 1 or date.month == 7:  # Show month and year for January
        return date.strftime('%b\n%Y')
    else:  # Show only month for other months
        return date.strftime('%b')


def plot_pv_counter(df: pd.DataFrame, scale: float) -> None:
    """
    Plot a graph for 'pv' values from the DataFrame.

    :param df: DataFrame containing the 'pv' values. Expects a column named 'pv'.
    :param scale: Scaling factor for PV supply.
    """
    path = 'plot'
    if not os.path.exists(path):
        os.makedirs(path)
    df = df.to_frame()
    # df = local_time_to_utc(df)

    df = interpolate_data(df)

    df = df.resample('H').mean()
    pv = (scale * df / 1e3).values

    fig, ax = plt.subplots()

    # Plotting 'pv' on the first y-axis
    ax.plot(df.index, pv, color='orange', linewidth=0.8, alpha=1)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Power [kW]')
    ax.tick_params(axis='y')

    # Set major and minor locators for the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))  # Set minor ticks in the middle of the month

    # Set formatter for major ticks
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_month_year_formatter))

    ax.tick_params(axis='both', which='minor')

    plt.xlim(df.index[0], df.index[-1] + pd.Timedelta(hours=1))
    plt.ylim(0, scale * MAX_PV_POWER / 1e3)
    plt.title('PV Power', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{path}/pv.png')
    plt.show()


def main():
    df_pv = process_zl()
    df_pv.to_csv('pv_supply.csv', sep=';')

    df_pv = pd.read_csv('pv_supply.csv', sep=';', index_col=0)['pv']
    df_pv.index = pd.to_datetime(df_pv.index)

    plot_pv_counter(df_pv, scale=0.25)  # for scale see scenario.json


if __name__ == '__main__':
    main()
