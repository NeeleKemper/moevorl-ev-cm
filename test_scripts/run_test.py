import argparse
import os

import pandas as pd
import numpy as np
from simulation.utils import make_test_env
from morl.multi_policy.evorl.evorl_neat import EvoRLNEAT
from morl.multi_policy.evorl.evorl_policy_net import EvoRLPolicyNet
from morl.multi_policy.mo_ddpg.mo_ddpg import MODDPG


def run_episode(agent, env, w):
    r = np.array([0, 0, 0])
    obs, episode_done = env.reset()[0], False
    while not episode_done:
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
        episode_done = terminated or truncated
    if r[0] > 0:
        r = r / 100
        success = 1
    else:
        success = 0
    return np.round(r, 4), success


def create_df():
    rewards_columns = ['agent_id', 'env_id', 'weight_number', 'env_number', 'r0', 'r1', 'r2',
                       'success']  # use success to calculate success_rate
    soc_metrics_columns = ['soc_mean', 'soc_std', 'soc_cov', 'soc_min', 'soc_max', 'charged', 'charged_amount_mean',
                           'target_reached', 'target_diff_mean']
    pv_metrics_columns = ['self_consumed_mean', 'ext_mean', 'feed_mean', 'self_consumption_rate', 'feed_back_rate',
                          'grid_dependency']
    load_metrics_columns = ['papr', 'load_std', 'load_mean', 'load_sum', 'max_load', 'load_factor', 'lvi']
    constraint_metrics_columns = ['time_step', 'l1', 'l2', 'l3', 'ext_grid', 'phase_diff_l1_l2',
                                  'phase_diff_l1_l3', 'phase_diff_l2_l3', 'grid_diff']

    df = pd.DataFrame(columns=rewards_columns + soc_metrics_columns + pv_metrics_columns + load_metrics_columns +
                              constraint_metrics_columns)
    return df


def load_setup(algorithm: str, scenario: str,  data_set: str, agent_seed: int, env_seed: int,
               test_config: dict):
    charging_density = test_config['charging_density']
    pv_noise = test_config['pv_noise']
    model_name = f'{algorithm.upper()}_{scenario}_seed_{agent_seed}'

    envs = make_test_env(scenario=f'{scenario}', data_set=data_set,
                         multiple_envs=True, charging_density=charging_density,
                         pv_noise=pv_noise, seed=env_seed)

    if algorithm == 'moddpg':
        agent = MODDPG(envs, net_arch=[1024, 1024, 1024], log=False)
    elif algorithm == 'ff_neat' or algorithm == 'rnn_neat':
        agent = EvoRLNEAT(envs, algorithm=algorithm, log=False)
    elif algorithm == 'ff_spea2':
        if scenario == 'scenario_CS05' or scenario == 'scenario_CS10' or scenario == 'scenario_CS15':
            agent = EvoRLPolicyNet(envs, network_type='FF', algorithm='SPEA2', net_arch=[32, 32, 32], log=False)
        else:
            agent = EvoRLPolicyNet(envs, network_type='FF', algorithm='SPEA2', net_arch=[32, 32, 32, 32], log=False)
    elif algorithm == 'ff_nsga2':
        agent = EvoRLPolicyNet(envs, network_type='FF', algorithm='NSGA2', net_arch=[32, 32, 32], log=False)
    else:  # algorithm == 'lstm_nsga':
        if scenario == 'scenario_CS15':
            agent = EvoRLPolicyNet(envs, network_type='LSTM', algorithm='NSGA2', rnn_hidden_size=8, rnn_num_layers=4,
                                   log=False)
        else:
            agent = EvoRLPolicyNet(envs, network_type='LSTM', algorithm='NSGA2', rnn_hidden_size=8, rnn_num_layers=3,
                                   log=False)

    agent.load(sub_folder=scenario, filename=model_name)
    weights = agent.get_weights()
    df = create_df()
    return envs, agent, weights, df


def test_run(envs, agent, weights, df, agent_id: int = 42, env_id: int = 42):
    df_failures = pd.DataFrame()
    for i, w in enumerate(weights):
        for j, env in enumerate(envs):
            reward, success = run_episode(agent, env, w)
            if success:
                results = env.calculate_metrics()
                metrics_list = list(results.values())
                constraint_metrics_list = [-1] * 9
            else:
                metrics_list = [-1] * 22
                results, df_obs = env.calculate_constraint_metrics()
                df_failures = pd.concat([df_failures, df_obs], ignore_index=True)
                constraint_metrics_list = list(results.values())
            df.loc[len(df)] = [agent_id, env_id, i, j, reward[0], reward[1], reward[2],
                               success] + metrics_list + constraint_metrics_list

    r0 = np.round(np.mean(df['r0'].to_list()), 2)
    r1 = np.round(np.mean(df['r1'].to_list()), 2)
    r2 = np.round(np.mean(df['r2'].to_list()), 2)
    success = np.round(np.mean(df['success'].to_list()), 2)
    print(f'Agent {agent_id} -> R: [{r0}, {r1}, {r2}]; success: {success}')
    return df, df_failures


def test_utilization(algorithm: str, scenario: str, test_config: dict, agent_seed: int = 42,
                     env_seed: int = 43):
    df_results, df_ev_info_results, df_failure_results = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    charging_density = test_config['charging_density']

    envs, agent, weights, df = load_setup(algorithm=algorithm, scenario=scenario, data_set='all',
                                          agent_seed=agent_seed, env_seed=env_seed, test_config=test_config)

    df, df_failure = test_run(envs, agent, weights, df, agent_seed, env_seed)
    df = df.round(4)
    df_failure = df_failure.round(4)
    df_results = pd.concat([df_results, df], ignore_index=True)
    df_failure_results = pd.concat([df_failure_results, df_failure], ignore_index=True)
    # save results as csv
    model = f'{algorithm.upper()}_{scenario}_agent_{agent_seed}_env_{env_seed}'
    path = f'results/test_utilization/{algorithm}/{scenario}_{charging_density}'
    if not os.path.isdir(path):
        os.makedirs(path)
    # strange bug: Rewards are only saved rounded by the following line.
    df_results.to_csv(f'{path}/{model}.csv', sep=';', index=False)
    df_failure_results.to_csv(f'{path}/{model}_failures.csv', sep=';', index=False)
    print(f'Done: {model}')


def test_hold_out(algorithm, scenario, test_config, agent_seed=42, env_seed=42):

    envs, agent, weights, df = load_setup(algorithm=algorithm, scenario=scenario, data_set='test',
                                          agent_seed=agent_seed, env_seed=env_seed, test_config=test_config)
    df, df_failure = test_run(envs, agent, weights, df, agent_seed, env_seed)
    df = df.round(4)
    df_failure = df_failure.round(4)
    model = f'{algorithm.upper()}_{scenario}_agent_{agent_seed}_env_{env_seed}'
    path = f'results/test_hold_out/{algorithm}/{scenario}'
    if not os.path.isdir(path):
        os.makedirs(path)
    # strange bug: Rewards are only saved rounded by the following line.
    df.to_csv(f'{path}/{model}.csv', sep=';', index=False)
    df_failure.to_csv(f'{path}/{model}_failures.csv', sep=';', index=False)
    print(f'Done: {model}')


def main():
    parser = argparse.ArgumentParser(description='Run multi-agent or single-agent simulation.')
    parser.add_argument('--algorithm', type=str, default='ff_neat',
                        choices=['ff_neat', 'rnn_neat', 'ff_spea2', 'ff_nsga2', 'lstm_nsga2', 'moddpg'],
                        help='Algorithm to use')
    parser.add_argument('--scenario', type=str, default='scenario_CS05', help='Scenario to run')
    parser.add_argument('--hold_out', action='store_true', help='Flag to run in hold_out mode')
    parser.add_argument('--charging_density', type=str, default='norm', choices=['norm', 'high'],
                        help='Charging event density for event generation')
    parser.add_argument('--agent_seed', type=int, default=42, help='Seed of the agent used for training')
    args = parser.parse_args()

    hold_out = args.hold_out

    test_config = {
        'charging_density': args.charging_density,
        'pv_noise': not hold_out,
    }

    if hold_out:
        test_hold_out(args.algorithm, args.scenario, test_config, agent_seed=args.agent_seed, env_seed=42)
    else:
        test_utilization(args.algorithm, args.scenario, test_config, agent_seed=args.agent_seed, env_seed=71)


if __name__ == "__main__":
    main()
