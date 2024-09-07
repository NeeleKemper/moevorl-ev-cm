import glob
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from simulation.constant import MIN_BATTERY_CAP, MAX_BATTERY_CAP
from test_scripts.test_utils import get_scenario_name, ALGORITHMS_NAMES, ALGORITHMS, SCENARIOS, UTILIZATION, FAILURE_WINDOW_SIZE

sns.set()
warnings.filterwarnings("ignore")


def inverse_normalize(normalized_value, value_min, value_max):
    """
    Inverse of __normalize.
    Converts a normalized value back to its original scale.
    """
    return normalized_value * (value_max - value_min) + value_min


def map_ev(battery_capacity: float, electric_vehicles: {}):
    battery_capacity = round(inverse_normalize(battery_capacity, MIN_BATTERY_CAP - 100, MAX_BATTERY_CAP) / 1000, 2)
    for ev in electric_vehicles:
        model_data = electric_vehicles[ev]
        if model_data['battery_capacity_kwh'] == battery_capacity:
            return ev, electric_vehicles[ev]


def analyze_phase_difference(window: pd.DataFrame, failure_index: int, electric_vehicles: {}):
    df_failure = window.drop('failure', axis=1)
    # Assuming 'external_cs_conditions' is a list of columns to check
    # Define the threshold
    threshold = 0.99
    # Calculate the difference
    diff = df_failure.diff()

    # Identify where the change is greater than the threshold
    drastic_shifts = diff.abs() > threshold

    # Find indices where drastic shifts occur
    shift_indices = drastic_shifts.any(axis=1)

    # Find the last index of a drastic shift
    last_shift_idx = np.where(shift_indices)[0][-1] if np.any(shift_indices) else None

    if last_shift_idx is not None:
        if shift_indices.index[last_shift_idx] - failure_index == 0:
            try:
                last_shift_idx = np.where(shift_indices)[0][-2] if np.any(shift_indices) else None
            except:
                last_shift_idx = None
    failure_obs = []
    if last_shift_idx is not None:
        idx = shift_indices.index[last_shift_idx]
        shifted_columns = drastic_shifts.columns[drastic_shifts.loc[idx].values].tolist()

        for col in shifted_columns:
            cs = col.split('_')[-1]
            columns = [f'mod', f'soc_{cs}', f'battery_capacity_{cs}', f'charging_power_{cs}',
                       f'charging_time_{cs}', f'connected_phases_{cs}']

            if idx > 0 > diff.at[idx, col]:  # Negative shift, take the previous row
                desired_row = window.loc[idx - 1, columns]
                arrival_type = 'departure'
            else:  # Positive shift or first row, take the current row

                desired_row = window.loc[idx, columns]
                arrival_type = 'arrival'

            # Calculate the number of timesteps from the shift to the failure
            timesteps_to_failure = failure_index - idx
            ev, data = map_ev(desired_row[2], electric_vehicles)
            phases = data['n_charging_phases']
            mod = int(inverse_normalize(desired_row[0], 0, (24 * 60) - 1))
            failure_obs = [arrival_type, timesteps_to_failure, mod, int(cs), ev, phases]
    return failure_obs


def get_failure_type(df, n_charging_stations):
    phase_span, grid_power_span = 24, 1000 * n_charging_stations
    max_phase_difference = 20
    max_grid_power = 40000

    ext_grid = round(df['ext_grid'].iloc[-1] * (max_grid_power + grid_power_span), 2)
    l1_l2 = round(df['l1_l2'].iloc[-1] * (max_phase_difference + phase_span), 2)
    l1_l3 = round(df['l1_l3'].iloc[-1] * (max_phase_difference + phase_span), 2)
    l2_l3 = round(df['l2_l3'].iloc[-1] * (max_phase_difference + phase_span), 2)
    failure_types = []
    phase_imbalance = []
    grid_overload = []
    if l1_l2 > max_phase_difference:
        failure_types.append('l1_l2')
        phase_imbalance.append(l1_l2 - max_phase_difference)
    if l1_l3 > max_phase_difference:
        failure_types.append('l1_l3')
        phase_imbalance.append(l1_l3 - max_phase_difference)
    if l2_l3 > max_phase_difference:
        failure_types.append('l2_l3')
        phase_imbalance.append(l2_l3 - max_phase_difference)
    if ext_grid > max_grid_power:
        failure_types.append('ext_grid')
        grid_overload.append(ext_grid - max_grid_power)
    return failure_types, phase_imbalance, grid_overload


def get_n_charging_station(scenario):
    return {'scenario_CS05': 5, 'scenario_CS10': 10}.get(scenario, 15)


def load_ev_config():
    with open('../simulation/electric_vehicles.json', 'r') as f:
        electric_vehicles = json.load(f)
    return electric_vehicles


def read_csv(algorithm: str, scenario: str, utilization: str):
    print(f'Reading {algorithm} {scenario} {utilization}')
    df = pd.DataFrame()
    for file in glob.glob(
            f'results/test_utilization/{algorithm}/scenario_{scenario}_{utilization}/*failures.csv'):
        df_temp = pd.read_csv(file, sep=';')
        df = pd.concat([df, df_temp], ignore_index=True)

    df_success_rate = pd.DataFrame()
    for path in glob.glob(f'results/test_utilization/{algorithm}/scenario_{scenario}_{utilization}/*_71.csv'):
        df_temp = pd.read_csv(path, sep=';')
        df_success_rate = pd.concat([df_success_rate, df_temp], ignore_index=True)

    success_rate = round(df_success_rate.mean()['success'], 3)
    return df, success_rate


def get_mod_bins(mod_list):
    # Assuming mod_list is a list of minutes in the day
    hour_bins = []
    for t in mod_list:
        hour = t // 60  # Integer division to get the hour
        hour_bins.append(hour)
    return hour_bins


def calculate_percentages(data_dict: {}):
    total = sum(data_dict.values())
    percentages = {key: (value / total) * 100 for key, value in data_dict.items()}
    return percentages


def analyze_failures(scenario: str, algorithm: str, utilization: str, window_size: int):
    n_charging_stations = get_n_charging_station(scenario)
    electric_vehicles = load_ev_config()

    failure_count = {'l1_l2': 0, 'l1_l3': 0, 'l2_l3': 0, 'ext_grid': 0}

    df, success_rate = read_csv(algorithm, scenario, utilization)

    failures = df[df['failure'] == 1]

    df_failure_obs = pd.DataFrame(
        columns=['event', 'time_step', 'mod', 'cs', 'ev', 'phase', 'phase_imbalance', 'grid_overload'])

    for failure_index, failure_row in failures.iterrows():

        window = df[failure_index - window_size:failure_index + 1]
        if window.empty:
            continue
        failure_types, phase_imbalance, grid_overload = get_failure_type(window, n_charging_stations)
        for fail in failure_types:
            failure_count[fail] = failure_count.get(fail, 0) + 1
            # arrival_type, timesteps_to_failure, mod, cs, ev, phases
            failure_obs = analyze_phase_difference(window, failure_index, electric_vehicles)

            if len(failure_obs) > 0:
                df_failure_obs.loc[len(df_failure_obs)] = failure_obs + [np.mean(phase_imbalance),
                                                                         np.mean(grid_overload)]
            else:
                df_failure_obs.loc[len(df_failure_obs)] = ['unknown', np.nan, np.nan, np.nan, np.nan,
                                                           np.nan, np.nan, np.mean(grid_overload)]

    return failure_count, df_failure_obs, success_rate


def main():
    # Headers for the table: 'Metric' followed by algorithm names
    headers = ['Metric'] + ALGORITHMS_NAMES

    for scenario in SCENARIOS:
        for utilization in UTILIZATION:
            # Initialize table_data with headers
            table_data = {metric: [] for metric in ['l1_l2', 'l1_l3', 'l2_l3', 'ext_grid',
                                                    'mean phase imbalance', 'std phase imbalance',
                                                    'max phase imbalance', 'min phase imbalance',
                                                    'mean grid overload', 'std grid overload', 'max grid overload',
                                                    'min grid overload',
                                                    'arrival', 'departure', 'unknown',
                                                    '1-phase', '2-phase', '3-phase',
                                                    'time_step: 1', 'time_step: 2', 'time_step: 3', 'time_step > 3']}

            for algorithm, algorithm_name in zip(ALGORITHMS, ALGORITHMS_NAMES):
                failure_count, df_failure_obs, success_rate = analyze_failures(scenario, algorithm, utilization,
                                                                               FAILURE_WINDOW_SIZE)
                counter = 0
                for key in failure_count:
                    counter += failure_count[key]
                print(scenario, utilization, algorithm, failure_count, counter, success_rate)
                continue
                df_failure_obs['mod_bins'] = get_mod_bins(df_failure_obs['mod'].tolist())

                # Calculating percentages
                class_columns = ['event', 'time_step', 'phase']
                failure_obs_percentages = {col: df_failure_obs[col].value_counts(normalize=True, dropna=True) * 100 for
                                           col in class_columns}
                failure_percentage = calculate_percentages(failure_count)

                # Adding data to table_data
                for key in failure_percentage:
                    table_data[key].append(f'{failure_percentage[key]:.2f}')

                for key in failure_obs_percentages:
                    if key == 'event':
                        for event in ['arrival', 'departure', 'unknown']:
                            if event in failure_obs_percentages[key]:
                                table_data[event].append(f'{failure_obs_percentages[key][event]:.2f}')
                            else:
                                table_data[event].append(0)
                    if key == 'phase':
                        for phase in [1, 2, 3]:
                            if phase in failure_obs_percentages[key]:
                                table_data[f'{phase}-phase'].append(f'{failure_obs_percentages[key][phase]:.2f}')
                            else:
                                table_data[f'{phase}-phase'].append(0)
                    if key == 'time_step':
                        time_sum = 0
                        for t in range(1, 15):
                            if t < 4:
                                if t in failure_obs_percentages[key]:
                                    table_data[f'time_step: {t}'].append(f'{failure_obs_percentages[key][t]:.2f}')
                                else:
                                    table_data[f'time_step: {t}'].append(0)
                            elif t in failure_obs_percentages[key]:
                                time_sum += failure_obs_percentages[key][t]
                        table_data['time_step > 3'].append(f'{time_sum:.2f}')

                mean_phase_imbalance = round(np.nanmean(df_failure_obs['phase_imbalance'].tolist()), 4)
                mean_grid_overload = round(np.nanmean(df_failure_obs['grid_overload'].tolist()), 4)
                std_phase_imbalance = round(np.nanstd(df_failure_obs['phase_imbalance'].tolist()), 4)
                std_grid_overload = round(np.nanstd(df_failure_obs['grid_overload'].tolist()), 4)
                max_phase_imbalance = round(np.nanmax(df_failure_obs['phase_imbalance'].tolist()), 4)
                max_grid_overload = round(np.nanmax(df_failure_obs['grid_overload'].tolist()), 4)
                min_phase_imbalance = round(np.nanmin(df_failure_obs['phase_imbalance'].tolist()), 4)
                min_grid_overload = round(np.nanmin(df_failure_obs['grid_overload'].tolist()), 4)
                table_data['mean phase imbalance'].append(mean_phase_imbalance)
                table_data['std phase imbalance'].append(std_phase_imbalance)
                table_data['max phase imbalance'].append(max_phase_imbalance)
                table_data['min phase imbalance'].append(min_phase_imbalance)

                table_data['mean grid overload'].append(mean_grid_overload)
                table_data['std grid overload'].append(std_grid_overload)
                table_data['max grid overload'].append(max_grid_overload)
                table_data['min grid overload'].append(min_grid_overload)

            # Convert table_data to a format suitable for tabulate
            formatted_table_data = [[metric] + values for metric, values in table_data.items()]
            print(f'\n{get_scenario_name(scenario, utilization, env_seed=71)}')
            print(tabulate(formatted_table_data, headers=headers, tablefmt='grid'))

    print('Done')


if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# import glob
#
#
#
#
# # Function to read a single text file and convert it to a DataFrame
# def read_table_from_text(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()[3:]  # Skip the first two lines with headers and separator
#         metric_names = []
#         data = []
#         for line in lines:
#             if '+' in line:
#                 continue
#             if line.strip():  # Ignore empty lines
#                 parts = line.split('|')
#                 parts = [p.strip() for p in parts]
#                 parts = list(filter(None, parts))
#                 metric = parts[0]
#                 values = [float(val) if val != 'nan' else np.nan for val in parts[1:]]
#                 metric_names.append(metric)
#                 data.append(values)
#
#         # Convert to DataFrame
#         df = pd.DataFrame(data, columns=['FF-NSGA-II', 'FF-SPEA2', 'LSTM-NSGA-II', 'FF-NEAT', 'RNN-NEAT', 'MODDPG'],
#                           index=metric_names)
#         return df
#
#
# # List all the text files
# file_paths = glob.glob("results/evaluation/failure_analysis/CS*")
#
# # Read all the tables into a list of DataFrames
# tables = [read_table_from_text(file) for file in file_paths]
#
# # Combine all the tables into a 3D numpy array (one layer per table)
# combined_data = np.array([df.values for df in tables])
#
# # Compute mean and standard deviation across the third axis (i.e., across all tables)
# mean_values = np.nanmean(combined_data, axis=0)
# std_values = np.nanstd(combined_data, axis=0)
#
# # Create DataFrame for mean and standard deviation
# mean_df = pd.DataFrame(mean_values, columns=tables[0].columns, index=tables[0].index).round(3)
# std_df = pd.DataFrame(std_values, columns=tables[0].columns, index=tables[0].index).round(3)
# print(mean_df.to_string())

