import wandb
import numpy as np
from simulation.utils import make_env
from morl.common.utils import reset_wandb_env
from morl.common.evaluation import seed_everything
from morl.multi_policy.evorl.evorl_neat import EvoRLNEAT

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
    'conn_add_prob': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.9,
    },
    'conn_delete_prob': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.4,
    },
    'node_add_prob': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.9,
    },
    'node_delete_prob': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.4,
    },
    'survival_threshold': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.3,
    },
    'aggregation_mutate_rate': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.3,
    },
    'weight_mutate_rate': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.9,
    },
    'bias_mutate_rate': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.9,
    },
    'batch_size': {
        'distribution': 'categorical',
        'values': [8, 16, 32]
    },
    'env_iterations': {
        'distribution': 'categorical',
        'values': [2, 5, 10]
    },
}
sweep_config['parameters'] = parameters_dict

seed = 1
scenario = 'scenario_CS05'

experiment_name = 'MORL-Tune'
project_name = f'Tune_RNN_NEAT_{scenario}'
pop_size = 200

env_list = make_env(scenario=scenario, data_set='train', multiple_envs=True, seed=seed)
eval_env_list = make_env(scenario=scenario, data_set='tune', multiple_envs=True, seed=seed)

ref_point = np.array([-2.1, -2.1, -2.1])


def train():
    sweep_run = wandb.init()

    config = wandb.config

    # Reset the wandb environment variables
    reset_wandb_env()

    pop_size = 200
    agent = EvoRLNEAT(
        env_list,
        algorithm='RNN_NEAT',
        config_file='rnn-neat-config',
        pop_size=pop_size,
        conn_add_prob=config.conn_add_prob,
        conn_delete_prob=config.conn_delete_prob,
        node_add_prob=config.node_add_prob,
        node_delete_prob=config.node_delete_prob,
        survival_threshold=config.survival_threshold,
        activation_mutate_rate=0.0,
        aggregation_mutate_rate=config.aggregation_mutate_rate,
        weight_mutate_rate=config.weight_mutate_rate,
        bias_mutate_rate=config.bias_mutate_rate,
        batch_size=config.batch_size,
        env_iterations=config.env_iterations,
        project_name=project_name,
        experiment_name=experiment_name,
        wandb_entity=None,
        log=True,
        seed=seed,
        device='auto',
        n_jobs=8,
        tune=True
    )

    agent.train(
        n_generations=200,
        ref_point=ref_point,
        eval_envs=eval_env_list,
        known_pareto_front=None,
        num_eval_episodes_for_front=1,
        eval_freq=50,
        hv_eval_freq=50,
        save_file_name='EvoRL_RNN_NEAT-Tune'
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
