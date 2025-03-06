import os
import re
import glob
import pandas as pd
import numpy as np

from test_scripts.test_utils import (calculate_rl_metrics, calculate_mo_metrics, calculate_object_metrics, UTILIZATION,
                                     LOG_METRICS, ALGORITHMS, SCENARIOS, get_scenario_name, generate_plots,
                                     evaluate_metrics)

PATH_UTILIZATION = '../results/evaluation/test_utilization'
PATH_HOLDOUT = '../results/evaluation/test_hold_out'


def read_csv(model: str, scenario: str, utilization: str = None, sub_path: str = 'test_utilization'):
    df = pd.DataFrame()
    pattern = re.compile(r".*_[0-9]+\.csv$")
    csv_pattern = f'../results/{sub_path}/{model}/scenario_{scenario}' + (
        f'_{utilization}' if utilization else '') + '/*.csv'
    csv_files = glob.glob(csv_pattern)
    filtered_files = [file for file in csv_files if pattern.match(file)]
    for path in filtered_files:
        df_temp = pd.read_csv(path, sep=';', index_col=None)
        df = pd.concat([df, df_temp], ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def process_agents(df: pd.DataFrame, df_other: pd.DataFrame = None):
    df_results = pd.DataFrame(columns=LOG_METRICS)
    df_agent = df.groupby('agent_id')
    df_agent_other = df_other.groupby('agent_id') if df_other is not None else None

    for (agent_id, df_group) in df_agent:
        rl_metrics = calculate_rl_metrics(df_group)
        mo_metrics = calculate_mo_metrics(df_group,
                                          df_agent_other.get_group(agent_id) if df_agent_other is not None else None)
        obj_metrics = calculate_object_metrics(df_group)

        metrics = list(rl_metrics) + list(mo_metrics) + list(obj_metrics)
        df_results.loc[len(df_results)] = metrics
    return df_results


def calculate_metrics(utilization: bool = True):
    path = PATH_UTILIZATION if utilization else PATH_HOLDOUT
    sub_path = 'test_utilization' if utilization else 'test_hold_out'
    seed = 71 if utilization else 42
    for scenario in SCENARIOS:
        title_scenario = get_scenario_name(scenario, env_seed=seed)
        for algorithm in ALGORITHMS:
            print(f'\nAlgorithm: {algorithm} - Scenario: {scenario}')
            if utilization:
                for util in UTILIZATION:
                    df = read_csv(model=algorithm, scenario=scenario, utilization=util, sub_path=sub_path)
                    df_results = process_agents(df)
                    title_util_scenario = get_scenario_name(scenario, util, 71)
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    df_results.to_csv(f'{path}/{algorithm.upper()}_{title_util_scenario}.csv', sep=';', index=False)
            else:
                df = read_csv(model=algorithm, scenario=scenario, sub_path=sub_path)
                df_other = read_csv(model=algorithm, scenario=scenario, sub_path=sub_path)
                df_results = process_agents(df, df_other)
                if not os.path.isdir(path):
                    os.makedirs(path)
                df_results.to_csv(f'{path}/{algorithm.upper()}_{title_scenario}.csv', sep=';', index=False)


def main():
    # Utilization evaluation
    # calculate_metrics(utilization=True)
    for scenario in SCENARIOS:
        for util in UTILIZATION:
            print(f'\n{get_scenario_name(scenario, util, 71)}')
            evaluate_metrics(PATH_UTILIZATION, scenario, utilization=util, env_seed=71)
            generate_plots(PATH_UTILIZATION, scenario, utilization=util, env_seed=71)
    #
    # # Hold-out evaluation
    # calculate_metrics(utilization=False)
    for scenario in SCENARIOS:
        print(f'\n{get_scenario_name(scenario)}')
        evaluate_metrics(PATH_HOLDOUT, scenario, utilization='norm', env_seed=42)
        generate_plots(PATH_HOLDOUT, scenario, utilization='norm', env_seed=42)


if __name__ == "__main__":
    main()
