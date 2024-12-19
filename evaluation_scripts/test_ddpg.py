import os
import re
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

from morl.common.pareto import filter_pareto_dominated
from test_scripts.test_utils import (calculate_rl_metrics, calculate_mo_metrics, calculate_object_metrics, UTILIZATION,
                                     LOG_METRICS,
                                     ALGORITHMS, SCENARIOS, get_scenario_name, print_metrics_table, METRICS,
                                     generate_plots, ALGORITHMS_NAMES)

sns.set(style="whitegrid", color_codes=True)

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


def process_agents(df: pd.DataFrame):
    df_results = pd.DataFrame(columns=['r0', 'r1', 'r2', 'success_rate'])
    df_agent = df.groupby('agent_id')

    for (agent_id, df_group) in df_agent:
        rl_metrics = calculate_rl_metrics(df_group)
        metrics = list(rl_metrics)
        df_results.loc[len(df_results)] = metrics
    return df_results


def calculate_metrics(utilization: bool = True):
    algorithm = 'ddpg'
    path = PATH_UTILIZATION if utilization else PATH_HOLDOUT
    sub_path = 'test_utilization' if utilization else 'test_hold_out'

    for scenario in SCENARIOS:
        title_scenario = get_scenario_name(scenario)
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
            df_results = process_agents(df)

            if not os.path.isdir(path):
                os.makedirs(path)
            df_results.to_csv(f'{path}/{algorithm.upper()}_{title_scenario}.csv', sep=';', index=False)


def evaluate_metrics(path: str, scenario: str, utilization: str = 'norm', env_seed: int = 42):
    algorithm = 'ddpg'
    metrics = ['r0', 'r1', 'r2', 'success_rate']
    title_scenario = get_scenario_name(scenario, utilization, env_seed)

    # Read the CSV file
    df = pd.read_csv(f'{path}/{algorithm.upper().replace("_", "-")}_{title_scenario}.csv',
                     sep=';')

    # Calculate mean and standard deviation
    df_mean = df.mean(axis=0)
    df_std = df.std(axis=0)

    # Prepare the table of metrics
    results = [
        [metric, f'{mean:.3f}', f'{std:.3f}']
        for metric, mean, std in zip(metrics, df_mean, df_std)
    ]
    print(tabulate(results, headers=['Metric', 'Mean', 'Std'], tablefmt='grid'))

    # Find the row with the highest success rate
    max_success_idx = df['success_rate'].idxmax()
    max_success_values = df.loc[max_success_idx, ['r0', 'r1', 'r2', 'success_rate']]

    # Print the results
    print("\nValues where success_rate is highest:")
    print(f"r0: {max_success_values['r0']:.3f}, "
          f"r1: {max_success_values['r1']:.3f}, "
          f"r2: {max_success_values['r2']:.3f}, "
          f"success_rate: {max_success_values['success_rate']:.3f}")

    # Plot the distributions
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    font_size = 10
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    axes = axes.flatten()
    metric_names = [r'$\mathbf{R_{SoC}}$', r'$\mathbf{R_{smooth}}$', r'$\mathbf{R_{PV}}$', 'Success Rate']
    bins = 8
    for i, metric in enumerate(metrics):
        if metric == 'success_rate':
            # sns.kdeplot(df[metric], ax=axes[i], fill=True, alpha=0.8, clip=(0, 1))
            axes[i].hist(df[metric], bins=bins, alpha=0.8)
            axes[i].set_xlim(0, 1)
            axes[i].set_xlabel(f'{metric_names[i]}', fontweight='bold', fontsize=font_size)
        else:
            axes[i].hist(df[metric], bins=bins, alpha=0.8)
            # axes[i].set_xlim(-1.5, 0.5)
            axes[i].set_xticks([-1, 0.0])
            axes[i].set_xlabel(f'{metric_names[i]}', fontweight='bold', math_fontfamily='dejavuserif',
                               fontsize=font_size + 2)
            # sns.kdeplot(df[metric], ax=axes[i], fill=True, alpha=0.8)
        if i == 0:  # Only set the y-label for the first plot
            # axes[i].set_ylabel('Density', fontweight='bold', fontsize=font_size)
            axes[i].set_ylabel('Frequency', fontweight='bold', fontsize=font_size)

        else:
            axes[i].set_ylabel('')  # Remove the y-label for other plots

    # Add a title over all subplots
    # fig.suptitle('Distribution of ', fontsize=font_size, fontweight='bold')  # Adjust y for spacing above the subplots

    plt.tight_layout()
    dir_box = f'{path}/plots/{title_scenario}/ddpg'
    if not os.path.isdir(dir_box):
        os.makedirs(dir_box)
    plt.savefig(f'{dir_box}/{title_scenario}.png')
    plt.show()
    plt.close()


def plot_pareto_front(path: str, scenario: str, utilization: str = 'norm', env_seed: int = 42, sub_path: str = None):
    # Define marker styles and colors for each algorithm
    markers = ['P', 's', 'D', 'v', '^', 'x']
    cmap = plt.colormaps['tab10']  # Use tab10 colormap
    colors = [cmap(i) for i in range(len(ALGORITHMS))]
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    font_size = 12
    s = 30

    title_scenario = get_scenario_name(scenario, utilization, env_seed)
    print(f'\nScenario: {scenario}')

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))  # Create 3 vertically stacked subplots

    # Process each algorithm
    for i, algorithm in enumerate(ALGORITHMS):
        print(f'Algorithm: {algorithm}')

        # Read and preprocess data
        if env_seed == 42:
            df = read_csv(model=algorithm, scenario=scenario, utilization=utilization)
        else:
            df = read_csv(model=algorithm, scenario=scenario, utilization=utilization, sub_path=sub_path)

        df_weight = df.groupby(['weight_number']).mean()
        solutions = df_weight[['r0', 'r1', 'r2']].values

        # Compute Pareto front
        pf = filter_pareto_dominated(solutions.tolist())

        # Split Pareto front into dimensions
        soc, load, pv = np.split(pf, 3, axis=1)

        # Plot Pareto fronts
        axes[0].scatter(soc, load, s=s, alpha=0.8,
                        color=colors[i % len(colors)], marker=markers[i % len(markers)])
        axes[1].scatter(soc, pv, s=s, alpha=0.8, color=colors[i % len(colors)],
                        marker=markers[i % len(markers)], label=f'{ALGORITHMS_NAMES[i]}')
        axes[2].scatter(pv, load, s=s, alpha=0.8, color=colors[i % len(colors)],
                        marker=markers[i % len(markers)])

    # Add baseline
    df = pd.read_csv(f'{path}/DDPG_{title_scenario}.csv', sep=';')
    baseline = df[['r0', 'r1', 'r2']].values
    soc_baseline, load_baseline, pv_baseline = np.split(baseline, 3, axis=1)

    idx_max_soc = np.argmax(soc_baseline)
    idx_max_load = np.argmax(load_baseline)
    idx_max_pv = np.argmax(pv_baseline)

    # Extrahiere die korrespondierenden Punkte
    soc_max, load_at_soc_max, pv_at_soc_max = soc_baseline[idx_max_soc].item(), load_baseline[idx_max_soc].item(), \
        pv_baseline[idx_max_soc].item()
    soc_at_load_max, load_max, pv_at_load_max = soc_baseline[idx_max_load].item(), load_baseline[idx_max_load].item(), \
        pv_baseline[idx_max_load].item()
    soc_at_pv_max, load_at_pv_max, pv_max = soc_baseline[idx_max_pv].item(), load_baseline[idx_max_pv].item(), \
        pv_baseline[idx_max_pv].item()

    # Punkte als Tupel zusammenstellen
    baseline_points = [
        (soc_max, load_at_soc_max, pv_at_soc_max),
        (soc_at_load_max, load_max, pv_at_load_max),
        (soc_at_pv_max, load_at_pv_max, pv_max)
    ]

    # Duplikate entfernen
    unique_baseline_points = list(set(baseline_points))

    # Entpacke die eindeutigen Punkte für die Plots
    soc_baselines, load_baselines, pv_baselines = zip(*unique_baseline_points)

    baseline_color = 'black'
    baseline_s = 30

    axes[0].scatter(soc_baselines, load_baselines, color=baseline_color,
                    s=baseline_s, alpha=0.5)
    axes[1].scatter(soc_baselines, pv_baselines, color=baseline_color,
                    s=baseline_s, label='DDPG', alpha=0.5)
    axes[2].scatter(pv_baselines, load_baselines, color=baseline_color,
                    s=baseline_s, alpha=0.5)

    for ax in axes:
        ax.grid(True)

    # Set legend only for the top plot
    # axes[1].legend(
    #    loc='center left',  # Legende links neben dem Ankerpunkt ausrichten
    #    bbox_to_anchor=(1, 0.5),  # Ankerpunkt: rechts außerhalb des Plots, zentriert in der Höhe
    #    fontsize=font_size
    # )
    fig.legend(
        loc='lower center',  # Position der Legende
        bbox_to_anchor=(0.5, 0.0),  # Zentriert unterhalb der Plots
        fontsize=font_size,
        ncol=3  # Anzahl der Spalten in der Legende
    )
    # Set axis labels for all plots

    axes[0].set_xlabel(r'$\mathbf{R_{SoC}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[0].set_ylabel(r'$\mathbf{R_{smooth}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[1].set_xlabel(r'$\mathbf{R_{SoC}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[1].set_ylabel(r'$\mathbf{R_{PV}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[2].set_xlabel(r'$\mathbf{R_{PV}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[2].set_ylabel(r'$\mathbf{R_{smooth}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # rect: [left, bottom, right, top]
    dir_box = f'{path}/plots/{title_scenario}/pareto_front'
    if not os.path.isdir(dir_box):
        os.makedirs(dir_box)
    plt.savefig(f'{dir_box}/{title_scenario}.png')
    plt.show()


def plot_pareto_front_single_agent(path: str, scenario: str, utilization: str = 'norm', env_seed: int = 42,
                                   sub_path: str = None):
    markers = ['P', 's', 'D', 'v', '^', 'x']
    cmap = plt.colormaps['tab10']  # Use tab10 colormap
    colors = [cmap(i) for i in range(len(ALGORITHMS))]
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    font_size = 12
    s = 30

    title_scenario = get_scenario_name(scenario, utilization, env_seed)
    print(f'\nScenario: {scenario}')

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))  # Create 3 vertically stacked subplots

    if env_seed == 42:
        df = read_csv(model='ddpg', scenario=scenario, utilization=utilization)
    else:
        df = read_csv(model='ddpg', scenario=scenario, utilization=utilization, sub_path=sub_path)
    df['mean_reward'] = df[['r0', 'r1', 'r2']].mean(axis=1)

    # Finde den Agenten mit dem höchsten durchschnittlichen Reward
    best_agent_id = df.groupby('agent_id')['mean_reward'].mean().idxmax()

    print(f"Agent with the highest mean reward: {best_agent_id}")

    # Filtere den DataFrame für den besten Agenten
    df = df[df['agent_id'] == best_agent_id]
    baseline = df[['r0', 'r1', 'r2']].mean().values
    soc_baseline, load_baseline, pv_baseline = baseline
    baseline_color = 'black'
    baseline_s = 40

    axes[0].scatter(soc_baseline, load_baseline, color=baseline_color,
                    s=baseline_s, alpha=0.5)
    axes[1].scatter(soc_baseline, pv_baseline, color=baseline_color,
                    s=baseline_s, label='DDPG', alpha=0.5)
    axes[2].scatter(pv_baseline, load_baseline, color=baseline_color,
                    s=baseline_s, alpha=0.5)

    # Process each algorithm
    for i, algorithm in enumerate(ALGORITHMS):
        print(f'Algorithm: {algorithm}')

        # Read and preprocess data
        if env_seed == 42:
            df = read_csv(model=algorithm, scenario=scenario, utilization=utilization)
        else:
            df = read_csv(model=algorithm, scenario=scenario, utilization=utilization, sub_path=sub_path)
        df = df[df['agent_id'] == best_agent_id]
        df_weight = df.groupby(['weight_number']).mean()
        solutions = df_weight[['r0', 'r1', 'r2']].values
        pf = filter_pareto_dominated(solutions.tolist())

        # Split Pareto front into dimensions
        pf_soc, pf_load, pf_pv = np.split(pf, 3, axis=1)
        soc, load, pv = np.split(solutions, 3, axis=1)

        # Plot Pareto fronts
        # axes[0].scatter(soc, load, s=10, alpha=0.5,
        #                color=colors[i % len(colors)], marker=markers[i % len(markers)])
        # axes[1].scatter(soc, pv, s=10, alpha=0.5, color=colors[i % len(colors)],
        #                marker=markers[i % len(markers)])
        # axes[2].scatter(pv, load, s=10, alpha=0.5, color=colors[i % len(colors)],
        #                marker=markers[i % len(markers)])

        axes[0].scatter(pf_soc, pf_load, s=s, alpha=0.8,
                        color=colors[i % len(colors)], marker=markers[i % len(markers)])
        axes[1].scatter(pf_soc, pf_pv, s=s, alpha=0.8, color=colors[i % len(colors)],
                        marker=markers[i % len(markers)], label=f'{ALGORITHMS_NAMES[i]}')
        axes[2].scatter(pf_pv, pf_load, s=s, alpha=0.8, color=colors[i % len(colors)],
                        marker=markers[i % len(markers)])

    # Set legend only for the top plot
    # axes[1].legend(
    #    loc='center left',  # Legende links neben dem Ankerpunkt ausrichten
    #    bbox_to_anchor=(1, 0.5),  # Ankerpunkt: rechts außerhalb des Plots, zentriert in der Höhe
    #    fontsize=font_size
    # )
    for ax in axes:
        ax.grid(True)

    fig.legend(
        loc='lower center',  # Position der Legende
        bbox_to_anchor=(0.5, 0.0),  # Zentriert unterhalb der Plots
        fontsize=font_size,
        ncol=4  # Anzahl der Spalten in der Legende
    )
    # Set axis labels for all plots
    axes[0].set_xlabel(r'$\mathbf{R_{SoC}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[0].set_ylabel(r'$\mathbf{R_{smooth}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[1].set_xlabel(r'$\mathbf{R_{SoC}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[1].set_ylabel(r'$\mathbf{R_{PV}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[2].set_xlabel(r'$\mathbf{R_{PV}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')
    axes[2].set_ylabel(r'$\mathbf{R_{smooth}}$', math_fontfamily='dejavuserif', fontsize=font_size, fontweight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    dir_box = f'{path}/plots/{title_scenario}/pareto_front'
    if not os.path.isdir(dir_box):
        os.makedirs(dir_box)
    plt.savefig(f'{dir_box}/agent_{best_agent_id}_{title_scenario}.png')
    plt.show()


def main():
    # calculate_metrics(utilization=True)
    # calculate_metrics(utilization=False)

    for scenario in SCENARIOS:
        for util in UTILIZATION:
            print(f'\n{get_scenario_name(scenario, util, 71)}')
            evaluate_metrics(PATH_UTILIZATION, scenario, utilization=util, env_seed=71)
            plot_pareto_front(PATH_UTILIZATION, scenario=scenario, utilization=util, env_seed=71,
                              sub_path='test_utilization')
            plot_pareto_front_single_agent(PATH_UTILIZATION, scenario=scenario, utilization=util, env_seed=71,
                                           sub_path='test_utilization')

    for scenario in SCENARIOS:
        print(f'\n{get_scenario_name(scenario)}')
        evaluate_metrics(PATH_HOLDOUT, scenario, utilization='norm', env_seed=42)
        plot_pareto_front(PATH_HOLDOUT, scenario=scenario, utilization='norm', env_seed=42, sub_path='test_hold_out')
        plot_pareto_front_single_agent(PATH_UTILIZATION, scenario=scenario, utilization='norm', env_seed=71,
                                       sub_path='test_utilization')


if __name__ == "__main__":
    main()
