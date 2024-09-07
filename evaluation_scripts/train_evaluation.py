import os
import re
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tabulate import tabulate
from test_scripts.test_utils import get_scenario_name, get_metric_name, ALGORITHMS, ALGORITHMS_NAMES, SCENARIOS

sns.set()


def read_csv(algorithm: str, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(f'results/train/{algorithm.upper()}_scenario_{scenario}_train.csv', sep=';')
    df = df.rename(columns={'Unnamed: 0': 'global_step'})
    return df


def split_df(df: pd.DataFrame, algorithm: str, scenario: str):
    def extract_suffix(col_name):
        return ''.join(filter(str.isdigit, col_name))

    def remove_numeric_suffix(col_name):
        return re.sub(r'_[0-9]+$', '', col_name)

    unique_suffixes = set(extract_suffix(col) for col in df.columns if extract_suffix(col))
    dfs = {}

    for suffix in unique_suffixes:
        # Select columns that end with the current suffix
        selected_columns = [col for col in df.columns if col.endswith(suffix)]
        # Create a DataFrame with these columns
        df_temp = df[selected_columns]
        df_temp = df_temp.rename(columns=remove_numeric_suffix)
        df_temp = df_temp.loc[~(df_temp == 0).all(axis=1)]
        if len(df_temp) > 0:
            if algorithm != 'moddpg':
                batch_size = 32
                if algorithm == 'ff_neat' and (scenario == '1a' or scenario == '2a'):
                    batch_size = 16
                df_temp['generation'] = df_temp['num_episodes'] / batch_size
            else:
                df_temp['generation'] = np.nan
            dfs[suffix] = df_temp
    return dfs


def get_final_result(df: pd.DataFrame, metrics: list):
    max_hv_idx = df['hypervolume'].idxmax()
    results = df.loc[max_hv_idx]
    return results[metrics].tolist()


def remove_erroneous_zeros(series):
    """
    Replace sequences of zeros (except the first one) with NaN in a Pandas Series.
    """
    for i in range(len(series)):
        if series.iloc[i] == 0:
            series.iloc[i] = float('nan')
    return series.interpolate()  # Interpolate to fill NaNs


def get_list_result(df: pd.DataFrame, metrics: list):

    max_hv_idx = df['hypervolume'].idxmax()

    df_results = df.loc[:max_hv_idx][metrics]

    # Remove erroneous zero sequences in specified metrics
    for metric in metrics:
        if metric in df_results.columns:
            df_results[metric] = remove_erroneous_zeros(df_results[metric])

    if df_results['num_episodes'].iloc[0] != 0:
        start_row = {'num_episodes': 0, 'r0': -2, 'r1': -2, 'r2': -2, 'success_rate': 0, 'hypervolume': 0,
                     'sparsity': 0, 'generation': 0}
        df_results = pd.concat([pd.DataFrame([start_row]), df_results], ignore_index=True)

    df_results = df_results.interpolate()

    df_results = df_results.set_index('num_episodes')
    return df_results



def plot_results(dfs_finals, scenario, metric):
    plt.figure(figsize=(10, 6))
    x_max = 0
    for i, dfs in enumerate(dfs_finals):
        # Concatenate all dataframes for the current algorithm
        df_concat = pd.DataFrame()

        for j, df in enumerate(dfs):
            # Rename columns with a suffix
            df_renamed = df.rename(columns={col: f'{col}_{j}' for col in df.columns})
            # Join renamed DataFrame
            df_concat = df_concat.join(df_renamed, how='outer')
        df_concat = df_concat.interpolate()
        df_concat = df_concat.clip(0, 100)
        # Calculate mean and std for each metric across rows
        metric_cols = [col for col in df_concat.columns if col.startswith(metric)]
        df_concat[f'{metric}_mean'] = df_concat[metric_cols].mean(axis=1)
        df_concat[f'{metric}_std'] = df_concat[metric_cols].std(axis=1)
        # Plot mean and margin (mean ± std)
        plt.plot(df_concat.index, df_concat[f'{metric}_mean'], label=ALGORITHMS_NAMES[i], linewidth=2)
        plt.fill_between(df_concat.index,
                         df_concat[f'{metric}_mean'] - df_concat[f'{metric}_std'],
                         df_concat[f'{metric}_mean'] + df_concat[f'{metric}_std'],
                         alpha=0.2)

        if x_max < max(df_concat.index):
            x_max = max(df_concat.index)
    plt.xlim([0, x_max + 100])

    title_scenario = get_scenario_name(scenario)
    title_metric = get_metric_name(metric)
    plt.title(f'{title_metric} Comparison: {title_scenario}', fontsize=14, fontweight='bold')
    plt.xlabel('Total Number of Episodes (Experiences)')
    plt.ylabel(title_metric)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid(True)

    path = f'results/evaluation/train/plots/{title_scenario}'
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{title_scenario}_{metric}_comparison.png')
    plt.show()


def main():
    metrics = ['r0', 'r1', 'r2', 'num_episodes', 'success_rate', 'hypervolume', 'sparsity', 'generation']
    for scenario in SCENARIOS:
        mean_values, std_values, all_dfs = [], [], []
        for algorithm in ALGORITHMS:
            df = read_csv(algorithm, scenario)
            dfs = split_df(df, algorithm, scenario)
            final_results = []
            dfs_finals = []
            for seed in range(42, 72):
                results = get_final_result(dfs[str(seed)], metrics)
                df_list = get_list_result(dfs[str(seed)], metrics)
                final_results.append(results)
                dfs_finals.append(df_list)
            final_results = np.array(final_results)
            all_dfs.append(dfs_finals)  # (pd.concat(dfs_finals, ignore_index=True))
            results_mean = np.round(np.mean(final_results, axis=0), 4)
            results_std = np.round(np.std(final_results, axis=0), 4)
            mean_values.append(results_mean)
            std_values.append(results_std)

        for metric in metrics:
            if metric != 'num_episodes' and metric != 'generation':
                plot_results(all_dfs, scenario, metric)

        # Preparing table data
        table_data = []
        for j, metric in enumerate(metrics):
            row = [metric]  # Start with the metric name
            for i in range(len(ALGORITHMS)):
                # Append the mean and std value for each algorithm
                if metric == 'num_episodes' or metric == 'generation':
                    cell_value = f'{mean_values[i][j]:.{0}f} ({std_values[i][j]:.{0}f})'
                else:
                    cell_value = f'{mean_values[i][j]:.{3}f} ({std_values[i][j]:.{3}f})'
                row.append(cell_value)
            table_data.append(row)

        # Displaying the table
        print(f'\nScenario {scenario}')
        headers = ['Metric'] + [alg.upper() for alg in ALGORITHMS]
        print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == '__main__':
    main()
