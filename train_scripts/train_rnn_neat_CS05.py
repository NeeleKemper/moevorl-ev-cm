import argparse
import numpy as np
from simulation.utils import make_env
from morl.multi_policy.evorl.evorl_neat import EvoRLNEAT


def evorl_rnn_neat(scenario: str, seed: int):
    project_name = f'RNN_NEAT_{scenario}'
    experiment_name = f'{project_name}_seed_{str(seed).zfill(2)}'
    env_list = make_env(scenario=scenario,  data_set='train', multiple_envs=True, seed=42)
    eval_env_list = make_env(scenario=scenario,  data_set='val',  multiple_envs=True, seed=42)
    ref_point = np.array([-2.1, -2.1, -2.1])

    pop_size = 200
    agent = EvoRLNEAT(
        # env (gym.Env) – The environment to train on.
        env_list,
        algorithm='RNN_NEAT',
        config_file='rnn-neat-config',
        pop_size=pop_size,
        conn_add_prob=0.72,
        conn_delete_prob=0.35,
        node_add_prob=0.66,
        node_delete_prob=0.16,
        survival_threshold=0.17,
        activation_mutate_rate=0.0,
        aggregation_mutate_rate=0.24,
        weight_mutate_rate=0.81,
        bias_mutate_rate=0.78,
        batch_size=32,
        env_iterations=5,
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
    parser = argparse.ArgumentParser(description='Run the EvoRL RNN-NEAT algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    evorl_rnn_neat(scenario='scenario_CS05', seed=args.seed)
