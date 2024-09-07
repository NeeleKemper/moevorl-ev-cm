import os
import pickle
import neat
import random
import torch
import time
import numpy as np
from joblib import Parallel, delayed
from typing import List, Optional, Union

from morl.common.evaluation import seed_everything
from morl.common.morl_algorithm import MOAgent, MOPolicy
from morl.common.evaluation import (
    log_all_multi_policy_metrics,
    policy_evaluation_evorl
)
from envs.ev_charging.ev_charging import EVCharging
from morl.multi_policy.evorl.utils import run_episode


class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates, hidden coordinates and a resolution defaulting to 10.0.
    """

    def __init__(self, input_coordinates, output_coordinates, hidden_coordinates=(), res=10.0):
        self.input_coordinates = input_coordinates
        self.hidden_coordinates = hidden_coordinates
        self.output_coordinates = output_coordinates
        self.res = res


def generate_input_coordinates(input_dim):
    return [(x / (input_dim - 1) * 2 - 1, -1) for x in range(input_dim)]


def generate_hidden_coordinates(layers: list):
    # Calculate the input dimension and then the sizes of hidden layers
    y_coords = np.linspace(-1, 1, len(layers) + 2)[1:-1]
    hidden_coordinates = []
    for l, y in zip(layers, y_coords):
        hidden_dim = [(x / (l - 1) * 2 - 1, y) for x in range(l)]
        hidden_coordinates.append(hidden_dim)

    return [i for i in hidden_coordinates]


def generate_output_coordinates(output_dim):
    return [(x / (output_dim - 1) * 2 - 1, 1) for x in range(output_dim)]


class EvoRLHyperNeat(MOAgent, MOPolicy):
    def __init__(self,
                 envs: List[EVCharging],
                 config_file: str = 'hyperneat-config',
                 net_arch: List = [256, 256],
                 pop_size: int = 50,
                 conn_add_prob: float = 0.9,
                 conn_delete_prob: float = 0.2,
                 node_add_prob: float = 0.9,
                 node_delete_prob: float = 0.2,
                 survival_threshold: float = 0.2,
                 activation_mutate_rate: float = 0.1,
                 aggregation_mutate_rate: float = 0.1,
                 weight_mutate_rate: float = 0.8,
                 bias_mutate_rate: float = 0.8,
                 batch_size: int = 1,
                 env_iterations: int = 5,
                 project_name: str = 'MORL',
                 experiment_name: str = 'EvoRLHyperNeat',
                 wandb_entity: Optional[str] = None,
                 log: bool = True,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = 'auto',
                 n_jobs: int = -1
                 ):
        if seed is not None:
            seed_everything(seed)
        self.rnd = random.Random(seed)
        MOAgent.__init__(self, envs[0], device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        config_path = f'morl_baselines/multi_policy/evorl/config/{config_file}'
        self.config_file = config_file
        self.pop_size = pop_size
        self.net_arch = net_arch
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self.survival_threshold = survival_threshold
        self.activation_mutate_rate = activation_mutate_rate
        self.aggregation_mutate_rate = aggregation_mutate_rate
        self.weight_mutate_rate = weight_mutate_rate
        self.bias_mutate_rate = bias_mutate_rate
        self.batch_size = batch_size
        self.env_iterations = env_iterations
        self.n_jobs = n_jobs

        self.modify_config_file(config_path, pop_size)
        self.config = neat.Config(neat.DefaultGenome, neat.nsga2.NSGA2Reproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        self.envs = envs
        self.pop = neat.Population(self.config)

        self.envs_batch = None
        self.best_genome = None

        input_coordinates = generate_input_coordinates(self.observation_dim)
        hidden_coordinates = generate_hidden_coordinates(net_arch)
        output_coordinates = generate_output_coordinates(self.action_dim)

        self.substrate = Substrate(input_coordinates=input_coordinates, output_coordinates=output_coordinates,
                                   hidden_coordinates=hidden_coordinates)
        self.activations = min(len(hidden_coordinates) + 10, 100)

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def train(
            self,
            n_generations: int,
            eval_envs: List[EVCharging],
            ref_point: np.ndarray,
            known_pareto_front: Optional[List[np.ndarray]] = None,
            num_eval_weights_for_front: int = 100,
            num_eval_episodes_for_front: int = 5,
            eval_freq: int = 50,
            hv_eval_freq: int = 100,
            save_file_name: str = 'EvoRL_hyperneat'
    ):
        assert eval_freq % self.env_iterations == 0, \
            f'The variable eval_freq must be a multiple of env_iterations. Got {eval_freq} and {self.env_iterations}'

        if self.log:
            self.register_additional_config({'ref_point': ref_point.tolist(), 'known_front': known_pareto_front})

        # Optimization using NSGA2 in chunks
        generations = 0
        while generations < n_generations:
            # train loop
            self.envs_batch = self.rnd.sample(self.envs, self.batch_size)

            self.best_genome, sorted_population = self.pop.run(self.eval_genomes, self.env_iterations+1)

            generations += self.env_iterations
            self.global_step = generations

            best_params = sorted_population[:num_eval_weights_for_front]
            if eval_envs is not None and self.log:
                if self.global_step % hv_eval_freq == 0:
                    # Evaluation
                    evals = [
                        policy_evaluation_evorl(self, eval_envs, ew, rep=num_eval_episodes_for_front) for ew in
                        best_params]

                    avg_scalarized_return = np.mean([eval[0] for eval in evals])
                    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
                    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
                    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
                    avg_constraint_violations = np.mean([eval[4] for eval in evals])

                    self.report(
                        avg_scalarized_return,
                        avg_scalarized_discounted_return,
                        avg_vec_return,
                        avg_disc_vec_return,
                        avg_constraint_violations,
                        log_name='eval',
                    )
                    front = [eval[2] for eval in evals]
                    log_all_multi_policy_metrics(
                        current_front=front,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        ref_front=known_pareto_front,
                    )

                    self.save(filename=save_file_name)
                elif self.global_step % eval_freq == 0:
                    self.policy_eval(eval_envs, weights=best_params[0], log=self.log,
                                     num_episodes=num_eval_episodes_for_front,
                                     log_name="eval")

        self.close_wandb()

    def eval(self, obs: np.ndarray, network: any) -> Union[int, np.ndarray]:
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        for _ in range(self.activations):
            o = network.activate(obs)
        return o

    def eval_genomes(self, genomes, config):
        # Evaluate each genome sequentially
        start_time = time.time()
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self.eval_genome)(genome) for genome_id, genome in genomes
        )
        # Update genomes with results
        for (genome_id, genome), fitness in zip(genomes, results):
            genome.fitness = fitness
        print('Execution time: ', time.time() - start_time)
        # for genome_id, genome in genomes:
        #    genome.fitness = self.eval_genome(genome)

    def eval_genome(self, genome):
        # Create network
        cppn = neat.nn.FeedForwardNetwork.create(genome, self.config)
        network = neat.hyperneat.create_phenotype_network(cppn, self.substrate, 'sigmoid')

        # Compute fitness values for each environment in the batch
        fitness_values = np.zeros((self.batch_size, self.reward_dim))
        for i, env in enumerate(self.envs_batch):
            fitness_values[i] = run_episode(env, network, self)

        # Aggregate fitness values (consider replacing with np.mean for average fitness)
        fitness_values = np.sum(fitness_values, axis=0) * (-1)

        # Return NSGA2Fitness object with aggregated fitness values
        return neat.nsga2.NSGA2Fitness(*fitness_values)

    def get_network(self, genome: any):
        cppn = neat.nn.FeedForwardNetwork.create(genome, self.config)
        network = neat.hyperneat.create_phenotype_network(cppn, self.substrate, 'sigmoid')
        return network

    def get_config(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'config_file': self.config_file,
            'net_arch': self.net_arch,
            'pop_size': self.pop_size,
            'conn_add_prob': self.conn_add_prob,
            'conn_delete_prob': self.conn_delete_prob,
            'node_add_prob': self.node_add_prob,
            'node_delete_prob': self.node_delete_prob,
            'survival_threshold': self.survival_threshold,
            'activation_mutate_rate': self.activation_mutate_rate,
            'aggregation_mutate_rate': self.aggregation_mutate_rate,
            'weight_mutate_rate': self.weight_mutate_rate,
            'bias_mutate_rate': self.bias_mutate_rate,
            'batch_size': self.batch_size,
            'env_iterations': self.env_iterations,
            'seed': self.seed
        }

    def modify_config_file(self, file_path, pop_size) -> None:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Loop through lines and modify the necessary lines
        for i, line in enumerate(lines):
            if line.strip().startswith('pop_size'):
                lines[i] = f'pop_size = {pop_size}\n'
            elif line.strip().startswith('conn_add_prob'):
                lines[i] = f'conn_add_prob = {self.conn_add_prob}\n'
            elif line.strip().startswith('conn_delete_prob'):
                lines[i] = f'conn_delete_prob = {self.conn_delete_prob}\n'
            elif line.strip().startswith('node_add_prob'):
                lines[i] = f'node_add_prob = {self.node_add_prob}\n'
            elif line.strip().startswith('node_delete_prob'):
                lines[i] = f'node_delete_prob = {self.node_delete_prob}\n'
            elif line.strip().startswith('survival_threshold'):
                lines[i] = f'survival_threshold = {self.survival_threshold}\n'
            elif line.strip().startswith('activation_mutate_rate'):
                lines[i] = f'activation_mutate_rate = {self.activation_mutate_rate}\n'
            elif line.strip().startswith('aggregation_mutate_rate'):
                lines[i] = f'aggregation_mutate_rate = {self.aggregation_mutate_rate}\n'
            elif line.strip().startswith('weight_mutate_rate'):
                lines[i] = f'weight_mutate_rate = {self.weight_mutate_rate}\n'
            elif line.strip().startswith('bias_mutate_rate'):
                lines[i] = f'bias_mutate_rate = {self.bias_mutate_rate}\n'

            # Open the file in write mode and overwrite with modified lines
        with open(file_path, 'w') as file:
            file.writelines(lines)

    def save(self, save_dir: str = 'weights/', filename: str = None):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}/{filename}', 'wb') as f:
            pickle.dump({
                'pop': self.pop,
                'config': self.config,
                'substrate': self.substrate,
                'best_genome': self.best_genome
            }, f)

    def load(self, save_dir: str = 'weights/', filename: str = None):
        if not os.path.exists(f'{save_dir}/{filename}'):
            raise ValueError(f'No saved model found at {save_dir}/{filename}')

        with open(f'{save_dir}/{filename}', 'rb') as f:
            checkpoint = pickle.load(f)
        self.pop = checkpoint['pop']
        self.config = checkpoint['config']
        self.substrate = checkpoint['substrate']
        self.best_genome = checkpoint['best_genome']

    def update(self) -> None:
        pass
