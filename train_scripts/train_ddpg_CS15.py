import argparse
from simulation.utils import make_env
from ddpg import DDPGAgent


def ddpg(scenario: str, seed: int):
    project_name = f'DDPG_{scenario}'
    experiment_name = f'{project_name}_seed_{str(seed).zfill(2)}'
    env_list = make_env(scenario=scenario,  data_set='train', multiple_envs=True, seed=42)
    eval_env_list = make_env(scenario=scenario,  data_set='val',  multiple_envs=True, seed=42)

    agent = DDPGAgent(
        env_list,
        learning_rate=5e-5,
        gamma=0.99,
        tau=0.03,
        buffer_size=1000000,
        net_arch=[1024, 1024, 1024],
        batch_size=512,
        learning_starts=7500,
        per_alpha=0.7,
        policy_frequency=20,
        env_iterations=1,
        project_name=project_name,
        experiment_name=experiment_name,
        wandb_entity=None,
        log=True,
        seed=seed,
        device='cuda'
    )
    agent.train(
        total_timesteps=int(1e8),
        eval_envs=eval_env_list,
        num_eval_episodes_for_front=1,
        eval_freq=50000,
        reset_num_timesteps=False,
        sub_folder=scenario,
        save_file_name=experiment_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the DDPG algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    args = parser.parse_args()
    ddpg(scenario='scenario_CS15', seed=args.seed)
