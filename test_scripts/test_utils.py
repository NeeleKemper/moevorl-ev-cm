import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate
from morl.common.weights import equally_spaced_weights
from morl.common.pareto import filter_pareto_dominated
from morl.common.performance_indicators import epsilon_metric, hypervolume, r2_indicator, expected_utility, \
    spread, spacing, sparsity

sns.set()
hv_ref_point = np.array([-2.1, -2.1, -2.1])
utopian_point = np.array([1, 1, 1])
reward_dim = 3
n_sample_weights = 200
weights_set = equally_spaced_weights(reward_dim, n_sample_weights)
PAPR_MIN, PAPR_MAX = 1, 6.3043  # 99 quantile
LVI_MIN, LVI_MAX = 0, 1.7187  # 99 quantile

LOG_METRICS = ['r0', 'r1', 'r2', 'success_rate',
               'hypervolume', 'r2_indicator', 'eum', 'spread', 'spacing', 'sparsity', 'epsilon',
               'hv_soc', 'sparsity_soc', 'spread_soc', 'spacing_soc', 'hv_pv', 'sparsity_pv', 'spread_pv', 'spacing_pv',
               'hv_load', 'sparsity_load', 'spread_load', 'spacing_load', 'hv_obj', 'sparsity_obj', 'spread_obj',
               'spacing_obj']

METRICS = ['r0', 'r1', 'r2', 'success_rate', 'hypervolume', 'r2_indicator', 'spread', 'spacing', 'sparsity']
METRICS = ['r0', 'r1', 'r2', 'success_rate', 'hypervolume', 'r2_indicator', 'spread', 'spacing', 'sparsity']

ALGORITHMS = ['ff_nsga2', 'ff_spea2', 'lstm_nsga2', 'ff_neat', 'rnn_neat', 'moddpg']
ALGORITHMS_NAMES = ['FF-NSGA-II', 'FF-SPEA2', 'LSTM-NSGA-II', 'FF-NEAT', 'RNN-NEAT', 'MODDPG']
SCENARIOS = ['CS05', 'CS10', 'CS15']
UTILIZATION = ['norm', 'high']
FAILURE_WINDOW_SIZE = 15


def process_load_values(df: pd.DataFrame):
    df = df.replace(0, np.nan)
    df.loc[:, 'papr'] = - (df['papr'].clip(PAPR_MIN, PAPR_MAX) - PAPR_MIN) / (PAPR_MAX - PAPR_MIN)
    df.loc[:, 'lvi'] = - (df['lvi'].clip(LVI_MIN, LVI_MAX) - LVI_MIN) / (LVI_MAX - LVI_MIN)
    return df


def calculate_rl_metrics(df_group: pd.DataFrame, group_by: str | None = 'weight_number'):
    if group_by is None:
        df_agent_weight = df_group
    else:
        df_agent_weight = df_group.groupby(group_by).mean()
    rewards = df_agent_weight[['r0', 'r1', 'r2']].mean()
    success_rate = df_agent_weight[['success']].mean()
    return rewards['r0'], rewards['r1'], rewards['r2'], success_rate['success']


def calculate_mo_metrics(df_group: pd.DataFrame, df_group_other: pd.DataFrame = None,
                         group_by: str | None = 'weight_number'):
    if group_by is None:
        df_agent_weight = df_group
    else:
        df_agent_weight = df_group.groupby(group_by).mean()
    # pareto front
    front = df_agent_weight[['r0', 'r1', 'r2']]
    pareto_front = list(filter_pareto_dominated(front))
    hv_value = hypervolume(hv_ref_point, pareto_front)
    r2_indicator_value = r2_indicator(pareto_front, weights_set, utopian_point)
    eum_value = expected_utility(pareto_front, weights_set=weights_set)

    spread_value = spread(pareto_front, utopian_point)
    spacing_value = spacing(pareto_front, utopian_point)
    sparsity_value = sparsity(pareto_front)

    if df_group_other is not None:
        if group_by is None:
            df_agent_weight_other = df_group_other
        else:
            df_agent_weight_other = df_group_other.groupby(group_by).mean()
        front_other = df_agent_weight_other[['r0', 'r1', 'r2']]
        pareto_front_q = list(filter_pareto_dominated(front_other))
        epsilon_value = epsilon_metric(pareto_front, pareto_front_q, n_objectives=reward_dim)
    else:
        epsilon_value = 0
    return hv_value, r2_indicator_value, eum_value, spread_value, spacing_value, sparsity_value, epsilon_value


def calculate_object_metrics(df_group: pd.DataFrame, group_by: str | None = 'weight_number'):
    if group_by is None:
        df_agent_weight_obj = df_group[df_group['success'] == 1]
    else:
        df_agent_weight_obj = df_group[df_group['success'] == 1].groupby(group_by).mean()

    df_soc = df_agent_weight_obj[['soc_mean', 'soc_cov', 'charged_amount_mean']]
    df_soc.loc[:, 'soc_cov'] = -df_soc['soc_cov']
    pareto_front_soc = list(filter_pareto_dominated(df_soc))
    if len(pareto_front_soc) > 0:
        hv_soc = hypervolume(ref_point=np.array([0, -1, 0]), points=pareto_front_soc)
        sparsity_soc = sparsity(pareto_front_soc)
        spread_soc = spread(pareto_front_soc, np.array([1, 0, 1]))
        spacing_soc = spacing(pareto_front_soc, np.array([1, 0, 1]))
    else:
        print('pareto_front_soc is zero')
        hv_soc, sparsity_soc, spread_soc, spacing_soc = np.nan, np.nan, np.nan, np.nan

    df_pv = df_agent_weight_obj[['self_consumption_rate', 'feed_back_rate', 'grid_dependency']]
    df_pv.loc[:, 'feed_back_rate'] = - df_pv['feed_back_rate']
    df_pv.loc[:, 'grid_dependency'] = -df_pv['grid_dependency']
    pareto_front_pv = list(filter_pareto_dominated(df_pv))
    if len(pareto_front_pv) > 0:
        hv_pv = hypervolume(ref_point=np.array([0, -1, -1]), points=pareto_front_pv)
        sparsity_pv = sparsity(pareto_front_pv)
        spread_pv = spread(pareto_front_pv, np.array([1, 0, 0]))
        spacing_pv = spacing(pareto_front_pv, np.array([1, 0, 0]))
    else:
        print('pareto_front_pv is zero')
        hv_pv, sparsity_pv, spread_pv, spacing_pv = np.nan, np.nan, np.nan, np.nan

    df_load = df_agent_weight_obj[['papr', 'load_factor', 'lvi']]
    df_load = process_load_values(df_load)
    pareto_front_load = list(filter_pareto_dominated(df_load))
    if len(pareto_front_load) > 0:
        hv_load = hypervolume(ref_point=np.array([-1, 0, -1]), points=pareto_front_load)
        sparsity_load = sparsity(pareto_front_load)
        spread_load = spread(pareto_front_load, np.array([0, 1, 0]))
        spacing_load = spacing(pareto_front_load, np.array([0, 1, 0]))
    else:
        print('pareto_front_load is zero')
        hv_load, sparsity_load, spread_load, spacing_load = np.nan, np.nan, np.nan, np.nan

    df_obj = pd.concat([df_soc[['soc_mean']], df_pv[['self_consumption_rate']], df_load[['papr']]], axis=1)
    pareto_front_obj = list(filter_pareto_dominated(df_obj))
    if len(pareto_front_obj) > 0:
        hv_obj = hypervolume(ref_point=np.array([0, 0, -1]), points=pareto_front_obj)
        sparsity_obj = sparsity(pareto_front_load)
        spread_obj = spread(pareto_front_load, np.array([1, 1, 0]))
        spacing_obj = spacing(pareto_front_load, np.array([1, 1, 0]))
    else:
        print('pareto_front_obj is zero')

        hv_obj, sparsity_obj, spread_obj, spacing_obj = np.nan, np.nan, np.nan, np.nan
    return hv_soc, sparsity_soc, spread_soc, spacing_soc, hv_pv, sparsity_pv, spread_pv, spacing_pv, hv_load, \
        sparsity_load, spread_load, spacing_load, hv_obj, sparsity_obj, spread_obj, spacing_obj


def get_scenario_name(scenario: str, utilization: str = 'norm', env_seed: int = 42):
    return f'{scenario}-{utilization.title()}-{env_seed}'



def get_metric_name(metric: str):
    replacements = {
        'Obj': 'Overall',
        'Hv': 'HV',
        'Soc': 'SoC',
        'Pv': 'PV',
        'Eum': 'EUM'
    }
    title_metric = metric.replace('_', ' ').title()
    for old, new in replacements.items():
        title_metric = title_metric.replace(old, new)
    return title_metric


def evaluate_metrics(path: str, scenario: str, utilization: str = 'norm', env_seed: int = 42):
    title_scenario = get_scenario_name(scenario, utilization, env_seed)
    print(f'\n{title_scenario}')
    metrics = []
    results = {alg: [] for alg in ALGORITHMS}

    for algorithm in ALGORITHMS:
        df_temp = pd.read_csv(f'{path}/{algorithm.upper()}_{title_scenario}.csv',
                              sep=';')
        if not metrics:
            metrics = df_temp.columns.to_list()

        df_mean = df_temp.mean(axis=0)
        df_std = df_temp.std(axis=0)
        results[algorithm] = [f'{mean:.3f} ({std:.3f})' for mean, std in zip(df_mean, df_std)]

    print_metrics_table(metrics, ALGORITHMS, results)


def print_metrics_table(metrics, algorithms, results):
    table = [[metric] + [results[alg][i] for alg in algorithms] for i, metric in enumerate(metrics) if
             metric in METRICS]
    headers = ['Metric'] + algorithms
    print(tabulate(table, headers=headers, tablefmt='grid'))


def load_pareto_front(df: pd.DataFrame):
    df_agent = df.groupby('agent_id')
    pareto_fronts = []
    for agent_id, df_group in df_agent:
        df_agent_weight = df_group.groupby('weight_number').mean()
        # pareto front
        front = df_agent_weight[['r0', 'r1', 'r2']]
        pareto_front = list(filter_pareto_dominated(front))
        pareto_fronts.append(pareto_front)
    return pareto_fronts


def generate_plots(path: str, scenario: str, utilization: str = 'norm', env_seed: int = 42):
    title_scenario = get_scenario_name(scenario, utilization, env_seed)

    print(f'Generate plots: Scenario {scenario}')
    all_data = []

    # Load data and add a column to identify the algorithm
    for algorithm in ALGORITHMS:
        df_temp = pd.read_csv(f'{path}/{algorithm.upper()}_{title_scenario}.csv', sep=';')
        df_temp['algorithm'] = algorithm
        all_data.append(df_temp)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data)

    # Melt the DataFrame to long format for seaborn
    long_df = combined_df.melt(id_vars=['algorithm'], var_name='metric', value_name='value')

    # Plotting each metric separately
    for metric in METRICS:
        data = long_df[long_df['metric'] == metric]
        # Violin Plot
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x='algorithm', y='value', data=data, inner='point')
        title_metric = get_metric_name(metric)
        plt.title(f'{title_metric} - Violin Plot: {title_scenario}', fontsize=16, fontweight='bold')
        ax.set_xticklabels(ALGORITHMS_NAMES, fontsize=12)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel(title_metric, fontsize=14)
        plt.tight_layout()
        dir_violin = f'{path}/plots/{title_scenario}/violin_plots'
        if not os.path.isdir(dir_violin):
            os.makedirs(dir_violin)
        plt.savefig(f'{dir_violin}/{title_scenario}_{metric}_violin.png')
        # plt.show()
        plt.close()

        # Box Plot
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='algorithm', y='value', data=data)
        plt.title(f'{title_metric} - Box Plot: {title_scenario}', fontsize=16, fontweight='bold')
        ax.set_xticklabels(ALGORITHMS_NAMES, fontsize=12)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel(title_metric, fontsize=14)
        plt.tight_layout()
        dir_box = f'{path}/plots/{title_scenario}/box_plots'
        if not os.path.isdir(dir_box):
            os.makedirs(dir_box)
        plt.savefig(f'{dir_box}/{title_scenario}_{metric}_box.png')
        plt.show()
        plt.close()

        # Density Plot
        plt.figure(figsize=(10, 6))
        ax = sns.kdeplot(data=data, x='value', hue='algorithm', fill=True, common_norm=False, alpha=0.5)
        plt.title(f'{title_metric} - Density Plot: {title_scenario}', fontsize=16, fontweight='bold')
        plt.xlabel(title_metric, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.tight_layout()
        dir_density = f'{path}/plots/{title_scenario}/density_plots'
        if not os.path.isdir(dir_density):
            os.makedirs(dir_density)
        plt.savefig(f'{dir_density}/{title_scenario}_{metric}_density.png')
        # plt.show()
        plt.close()



