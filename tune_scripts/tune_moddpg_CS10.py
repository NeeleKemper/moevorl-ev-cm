import wandb
import numpy as np
from simulation.utils import make_env
from morl.common.utils import reset_wandb_env
from morl.common.evaluation import seed_everything
from morl.multi_policy.mo_ddpg.mo_ddpg import MODDPG

# grid Search – Iterate over every combination of hyperparameter values.
# Very effective, but can be computationally costly.

# random Search – Select each new combination at random according to provided distributions. Surprisingly effective!

# bayesian Search – Create a probabilistic model of metric score as a function of the hyperparameters, and choose
# parameters with high probability of improving the metric. Works well for small numbers of continuous parameters
# but scales poorly.
sweep_config = {
    'method': 'bayes'  # grid , random, bayes
}

metric = {
    'name': 'eval/hypervolume',
    'goal': 'maximize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'gamma': {
        'distribution': 'uniform',
        'min': 0.95,
        'max': 0.9999,
    },
    'tau': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1,
    },
    'per_alpha': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 1.0,
    },
    'buffer_size': {
        'distribution': 'int_uniform',
        'min': int(1e5),
        'max': int(2e6),
    },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 1e-5,
        'max': 1e-4,
    },
    'learning_starts': {
        'distribution': 'int_uniform',
        'min': 1000,
        'max': 10000,
    },
    'batch_size': {
        'distribution': 'categorical',
        'values': [128, 256, 512]
    },
    'policy_frequency': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 20,
    }
}
sweep_config['parameters'] = parameters_dict

seed = 1
scenario = 'scenario_CS10'
project_name = f'Tune_MODDPG_{scenario}a'
experiment_name = 'MORL-Tune'

env_list = make_env(scenario=scenario, data_set='train', multiple_envs=True, seed=seed)
eval_env_list = make_env(scenario=scenario, data_set='tune', multiple_envs=True, seed=seed)

ref_point = np.array([-2.1, -2.1, -2.1])


def train():
    sweep_run = wandb.init()

    config = wandb.config

    # Reset the wandb environment variables
    reset_wandb_env()

    agent = MODDPG(
        env_list,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        buffer_size=config.buffer_size,
        net_arch=[1024, 1024, 1024],
        batch_size=config.batch_size,
        learning_starts=config.learning_starts,
        per_alpha=config.per_alpha,
        policy_frequency=config.policy_frequency,
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
        num_eval_weights_for_front=100,
        num_eval_episodes_for_front=1,  # 5
        eval_freq=100000,
        hv_eval_freq=100000,
        reset_num_timesteps=False,
        save_file_name='MODDPG-Tune',
    )

    wandb.finish()


def main():
    sweep_count = 20

    seed_everything(seed)

    # Set up the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)

    # Run the sweep agent
    wandb.agent(sweep_id, function=train, count=sweep_count)


if __name__ == '__main__':
    main()
