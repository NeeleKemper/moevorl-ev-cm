import argparse
import numpy as np
from simulation.utils import make_env
from morl.multi_policy.evorl.evorl_neat import EvoRLNEAT


def evorl_ff_neat(scenario: str, seed: int):
    project_name = f'FF_NEAT_{scenario}'
    experiment_name = f'{project_name}_seed_{str(seed).zfill(2)}'
    env_list = make_env(scenario=scenario,  data_set='train', multiple_envs=True, seed=42)
    eval_env_list = make_env(scenario=scenario,  data_set='val',  multiple_envs=True, seed=42)
    ref_point = np.array([-2.1, -2.1, -2.1])

    pop_size = 200
    agent = EvoRLNEAT(
        env_list,
        algorithm='FF_NEAT',
        config_file='ff-neat-config',
        pop_size=pop_size,
        conn_add_prob=0.79,
        conn_delete_prob=0.33,
        node_add_prob=0.70,
        node_delete_prob=0.28,
        survival_threshold=0.16,
        activation_mutate_rate=0.0,
        aggregation_mutate_rate=0.26,
        weight_mutate_rate=0.65,
        bias_mutate_rate=0.66,
        batch_size=16,
        env_iterations=10,
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
    parser = argparse.ArgumentParser(description='Run the EvoRL FF-NEAT algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    evorl_ff_neat(scenario='scenario_CS15', seed=args.seed)
