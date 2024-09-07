import wandb
import numpy as np
from simulation.utils import make_env
from morl.common.utils import reset_wandb_env
from morl.common.evaluation import seed_everything
from morl.multi_policy.evorl.evorl_policy_net import EvoRLPolicyNet

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
    'rnn_hidden_size': {
        'distribution': 'categorical',
        'values': [8, 16, 32]
    },
    'rnn_num_layers': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 4
    },
    'rnn_bidirectional': {
        'distribution': 'categorical',
        'values': [True, False]
    },
    'sbx_prob': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.95,
    },
    'sbx_eta': {
        'distribution': 'int_uniform',
        'min': 3,
        'max': 30
    },
    'mut_prob': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.3,
    },
    'mut_eta': {
        'distribution': 'int_uniform',
        'min': 3,
        'max': 30,
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
network_type = 'LSTM'
algorithm = 'NSGA2'

experiment_name = 'MORL-Tune'
project_name = f'Tune_{network_type}_{algorithm}_{scenario}'
pop_size = 200

env_list = make_env(scenario=scenario, data_set='train', multiple_envs=True, seed=seed)
eval_env_list = make_env(scenario=scenario, data_set='tune', multiple_envs=True, seed=seed)

ref_point = np.array([-2.1, -2.1, -2.1])

def train():
    sweep_run = wandb.init()

    config = wandb.config

    # Reset the wandb environment variables
    reset_wandb_env()

    agent = EvoRLPolicyNet(
        env_list,
        pop_size=pop_size,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_num_layers=config.rnn_num_layers,
        rnn_dropout=0,
        rnn_bidirectional=False,
        sbx_prob=config.sbx_prob,
        sbx_eta=config.sbx_eta,
        mut_prob=config.mut_prob,
        mut_eta=config.mut_eta,
        network_type=network_type,
        algorithm=algorithm,
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
        save_file_name=f'EvoRL_{network_type}_{algorithm}-Tune'
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
