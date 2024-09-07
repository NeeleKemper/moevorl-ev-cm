import argparse
import numpy as np
from simulation.utils import make_env
from morl.multi_policy.evorl.evorl_policy_net import EvoRLPolicyNet


def evorl_rnn_policy_net(scenario: str, seed: int):
    network_type = 'LSTM'
    algorithm = 'NSGA2'
    project_name = f'{network_type}_{algorithm}_{scenario}'
    experiment_name = f'{project_name}_seed_{str(seed).zfill(2)}'
    env_list = make_env(scenario=scenario,  data_set='train', multiple_envs=True, seed=42)
    eval_env_list = make_env(scenario=scenario,  data_set='val',  multiple_envs=True, seed=42)
    ref_point = np.array([-2.1, -2.1, -2.1])

    pop_size = 200
    agent = EvoRLPolicyNet(
        # env (gym.Env) – The environment to train on.
        env_list,
        rnn_hidden_size=8,
        rnn_num_layers=4,
        rnn_dropout=0,
        rnn_bidirectional=False,
        sbx_prob=0.7,
        sbx_eta=11,
        mut_prob=0.2,
        mut_eta=20,
        batch_size=32,
        env_iterations=10,
        network_type=network_type,
        algorithm=algorithm,
        pop_size=pop_size,
        project_name=project_name,
        experiment_name=experiment_name,
        wandb_entity=None,
        log=True,
        seed=seed,
        device='auto',
        n_jobs=8,
    )
    agent.train(
        n_generations=1000,
        ref_point=ref_point,
        eval_envs=eval_env_list,
        known_pareto_front=None,
        num_eval_episodes_for_front=1,
        eval_freq=10,
        hv_eval_freq=50,
        sub_folder=scenario,
        save_file_name=experiment_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the EvoRL LSTM-NSGA2 algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    evorl_rnn_policy_net(scenario='scenario_CS15', seed=args.seed)
