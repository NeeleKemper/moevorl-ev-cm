import argparse
import numpy as np
from simulation.utils import make_env
from morl.multi_policy.mo_ddpg.mo_ddpg import MODDPG


def moddpg(scenario: str, seed: int):
    project_name = f'MODDPG_{scenario}'
    experiment_name = f'{project_name}_seed_{str(seed).zfill(2)}'
    env_list = make_env(scenario=scenario,  data_set='train', multiple_envs=True, seed=42)
    eval_env_list = make_env(scenario=scenario,  data_set='val',  multiple_envs=True, seed=42)
    ref_point = np.array([-2.1, -2.1, -2.1])

    agent = MODDPG(
        env_list,
        learning_rate=1.1e-5,
        gamma=0.99,
        tau=0.04,
        buffer_size=1000000,
        net_arch=[1024, 1024, 1024],
        batch_size=256,
        learning_starts=7500,
        per_alpha=0.71,
        policy_frequency=13,
        env_iterations=1,
        project_name=project_name,
        experiment_name=experiment_name,
        wandb_entity=None,
        log=True,
        seed=seed,
        device='auto'
    )
    agent.train(
        total_timesteps=int(1e8),
        ref_point=ref_point,
        eval_envs=eval_env_list,
        known_pareto_front=None,
        num_eval_weights_for_front=200,
        num_eval_episodes_for_front=1,
        eval_freq=50000,
        hv_eval_freq=100000,
        reset_num_timesteps=False,
        sub_folder=scenario,
        save_file_name=experiment_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the MODDPG algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    moddpg(scenario='scenario_CS05', seed=args.seed)
