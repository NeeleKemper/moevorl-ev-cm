import argparse
import numpy as np
from simulation.utils import make_env
from morl.multi_policy.evorl.evorl_policy_net import EvoRLPolicyNet


def evorl_ff_policy_net(scenario: str, seed: int):
    network_type = 'FF'
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
        net_arch=[32, 32, 32],
        sbx_prob=0.90,
        sbx_eta=23,
        mut_prob=0.18,
        mut_eta=22,
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
    parser = argparse.ArgumentParser(description='Run the EvoRL FF-NSGA2 algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    evorl_ff_policy_net(scenario='scenario_CS05', seed=args.seed)
