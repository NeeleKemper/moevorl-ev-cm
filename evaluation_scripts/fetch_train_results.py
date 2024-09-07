import glob
import os

import numpy as np
import wandb
import pandas as pd
from test_scripts.test_utils import ALGORITHMS, SCENARIOS


def fetch_train_results(api: any, model: str, scenario: str, entity: str):
    project = f'{model}_scenario_{scenario}'

    runs = api.runs(entity + '/' + project)
    df_final = pd.DataFrame()
    seeds = []
    for run in runs:
        df = pd.DataFrame()
        global_step, num_episodes, training_time = [], [], []
        hv, eum, sparsity, fronts_path = [], [], [], []
        r0, r1, r2, success_rate = [], [], [], []
        seed = run.name.split('_')[-2]
        print(run.name)
        seeds.append(seed)
        for i, row in run.history().iterrows():
            global_step.append(row['global_step'])
            num_episodes.append(row['num_episodes'])
            training_time.append(row['training_time'])
            hv.append(row['eval/hypervolume'])
            eum.append(row['eval/eum'])
            sparsity.append(row['eval/sparsity'])

            r0.append(row['eval/vec_0'])
            r1.append(row['eval/vec_1'])
            r2.append(row['eval/vec_2'])

            success_rate.append(row['eval/success_rate'])

        df[f'global_step_{seed}'] = global_step
        df[f'num_episodes_{seed}'] = num_episodes
        df[f'training_time_{seed}'] = training_time
        df[f'hypervolume_{seed}'] = hv
        df[f'eum_{seed}'] = eum
        df[f'sparsity_{seed}'] = sparsity
        df[f'r0_{seed}'] = r0
        df[f'r1_{seed}'] = r1
        df[f'r2_{seed}'] = r2
        df[f'success_rate_{seed}'] = success_rate
        # group values
        df = df.groupby([f'global_step_{seed}']).max()
        # remove last row
        df = df[:-1]
        # interpolate:
        df = df.interpolate()
        df_final = pd.concat([df_final, df], axis=1)

    seeds.sort()
    print(f'{project}\nseeds ({len(seeds)}): {seeds}\n')
    path = '../results/train'
    if not os.path.isdir(path):
        os.makedirs(path)
    df_final.to_csv(f'{path}/{project}_train.csv', sep=';')


def filter_columns(df: pd.DataFrame, metrics: list):
    # Initialize an empty list to accumulate filtered columns
    all_filtered_columns = []

    # Loop over each metric and accumulate the filtered columns
    for metric in metrics:
        filtered_columns = [col for col in df.columns if
                            metric in col.strip() and '__MIN' not in col and '__MAX' not in col]
        all_filtered_columns.extend(filtered_columns)

    # Select only the accumulated filtered columns
    df_filtered = df[all_filtered_columns]

    return df_filtered


def rename_column(col):
    # Extract the seed number
    names = col.split('-')
    metrics = names[-1].split('/')[-1].strip()
    if metrics == 'vec_0': metrics = 'r0'
    if metrics == 'vec_1': metrics = 'r1'
    if metrics == 'vec_2': metrics = 'r2'
    seed_number = names[0].split('_')[-2]
    return f'{metrics}_{seed_number}'


def group_data(df: pd.DataFrame):
    indices = df.index.tolist()

    # Identify the breakpoints where a new group starts
    breakpoints = [0] + [i + 1 for i in range(len(indices) - 1) if indices[i + 1] - indices[i] > 1]
    # Add the last index
    breakpoints.append(len(indices))

    # Initialize an empty DataFrame for the summaries
    summarized_df = pd.DataFrame()

    # Iterate through each group and summarize
    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]

        # Summarize the rows for the current group
        summary_row = df.iloc[start_idx:end_idx].sum()
        # Append the summary row to the new DataFrame
        # Use the last index of the group as the index for the summary row
        summarized_df = pd.concat([summarized_df, pd.DataFrame([summary_row])])

    # Reset index if you want the index as a column
    summarized_df.reset_index(inplace=True)

    return summarized_df


def clean_column(col: pd.Series):
    # Find the index of the last non-NaN value
    last_non_nan_index = col.last_valid_index()

    # If a non-NaN value is found, replace all values from this index onwards with NaN
    if last_non_nan_index is not None:
        # Set to NaN from the last non-NaN value to the end of the series
        col.loc[last_non_nan_index:] = np.nan

    return col


def process_moddpg_files(scenario: str):
    project = f'MODDPG_scenario_{scenario}'
    metrics = ['num_episodes', 'success_rate', 'eval/vec_0', 'eval/vec_1', 'eval/vec_2', 'eval/hypervolume',
               'eval/sparsity']
    df_final = pd.DataFrame()
    for path in glob.glob(f'results/train/MODDPG_manuel_files/MODDPG_scenario_{scenario}*.csv'):
        df_temp = pd.read_csv(path, index_col='Step')
        df_temp = filter_columns(df_temp, metrics)
        df_temp.rename(columns={col: rename_column(col) for col in df_temp.columns}, inplace=True)
        for col in df_temp.columns:
            df_temp[col] = clean_column(df_temp[col])
        df_final = pd.concat([df_final, df_temp], axis=1).sort_index()
    df_final = group_data(df_final)
    path = '../results/train'
    if not os.path.isdir(path):
        os.makedirs(path)
    df_final.to_csv(f'{path}/{project}_train.csv', sep=';')


def main():
    api = wandb.Api()
    # for moddpg the history objective is broken and the files has to be downloaded manually
    # (see directory results/train/MODDPG_manuel_files)
    for algorithm in ALGORITHMS:
        for scenario in SCENARIOS:
            if algorithm == 'moddpg':
                process_moddpg_files(scenario)
            else:
                fetch_train_results(api, algorithm.upper(), scenario)


if __name__ == '__main__':
    main()
