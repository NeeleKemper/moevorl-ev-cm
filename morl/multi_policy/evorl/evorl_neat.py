import os
import pickle
import time
import wandb
import neat
import random
import torch

import numpy as np
from joblib import Parallel, delayed
from typing import List, Optional, Union

from morl.common.evaluation import seed_everything
from morl.common.morl_algorithm import MOAgent, MOPolicy
from morl.common.evaluation import (
    log_all_multi_policy_metrics,
    policy_evaluation_evorl,
    multi_policy_evaluation_evorl
)
from morl.multi_policy.evorl.utils import EarlyStopping
from envs.ev_charging.ev_charging import EVCharging
from envs.ev_charging_test.ev_charging_test import EVChargingTest
from morl.multi_policy.evorl.utils import run_episode


# example: https://neorl.readthedocs.io/en/latest/modules/neuroevolu/fneat.html

class EvoRLNEAT(MOAgent, MOPolicy):
    def __init__(self,
                 envs: List[EVCharging|EVChargingTest],
                 algorithm: str = 'FF_NEAT',
                 config_file: str = 'ff-neat-config',
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
                 experiment_name: str = 'EvoRL_FF_NEAT',
                 wandb_entity: Optional[str] = None,
                 log: bool = True,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = 'auto',
                 n_jobs: int = -1,
                 tune: bool = False
                 ):
        if seed is not None:
            seed_everything(seed)
        self.rnd = random.Random(seed)
        MOAgent.__init__(self, envs[0], device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        config_path = f'morl_baselines/multi_policy/evorl/config/{config_file}'
        self.config_file = config_file
        self.algorithm = algorithm
        self.pop_size = pop_size
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

        self.modify_config_file(config_path)
        self.config = neat.Config(neat.DefaultGenome, neat.nsga2.NSGA2Reproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        self.envs = envs
        self.pop = neat.Population(self.config, self.rnd)

        self.policy = None
        self.envs_batch = None
        self.genomes = None
        self.early_stopper = EarlyStopping(patience=10, restarts=0, min_delta=0)

        self.save_dir = f'models/{algorithm.lower()}'
        self.log = log
        self.tune = tune
        if self.log and not self.tune:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def train(
            self,
            n_generations: int,
            eval_envs: List[EVCharging],
            ref_point: np.ndarray,
            known_pareto_front: Optional[List[np.ndarray]] = None,
            num_eval_episodes_for_front: int = 5,
            eval_freq: int = 50,
            hv_eval_freq: int = 100,
            sub_folder: str = 'scenario_CS05',
            save_file_name: str = 'EvoRL_FF_NEAT'
    ):
        assert eval_freq % self.env_iterations == 0, \
            f'The variable eval_freq must be a multiple of env_iterations. Got {eval_freq} and {self.env_iterations}'

        if self.log:
            self.register_additional_config({'ref_point': ref_point.tolist(), 'known_front': known_pareto_front})

        # Optimization using NEAT in chunks
        generations = 0
        total_training_time = 0
        while generations < n_generations:
            # train loop
            self.envs_batch = self.rnd.sample(self.envs, self.batch_size)
            self.num_episodes += (self.batch_size * self.env_iterations)
            generation_start_time = time.time()
            _, self.genomes = self.pop.run(self.eval_genomes, self.env_iterations)
            generation_end_time = time.time()
            total_training_time += (generation_end_time - generation_start_time)

            generations += self.env_iterations
            self.global_step = generations

            if eval_envs is not None and self.log:
                if self.global_step % hv_eval_freq == 0:
                    avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return, \
                        avg_constraint_violations, front = \
                        multi_policy_evaluation_evorl(self, eval_envs, w=self.genomes, rep=num_eval_episodes_for_front)

                    self.report(
                        avg_scalarized_return,
                        avg_scalarized_discounted_return,
                        avg_vec_return,
                        avg_disc_vec_return,
                        avg_constraint_violations,
                        log_name='eval',
                    )
                    wandb.log({'num_episodes': self.num_episodes, 'training_time': round(total_training_time)})

                    log_all_multi_policy_metrics(
                        current_front=front,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        ref_front=known_pareto_front,
                    )

                    if not self.tune:
                        hypervolume = wandb.run.summary['eval/hypervolume']
                        if self.early_stopper(hypervolume, algorithm=self.algorithm, genomes=self.genomes):
                            if self.early_stopper.do_restart():
                                hv_eval_freq = int(hv_eval_freq / 2)
                                eval_freq = int(eval_freq / 2)
                                best_algorithm, best_genomes = self.early_stopper.load_best_model()
                                self.algorithm = best_algorithm if best_algorithm is not None else self.algorithm
                                self.genomes = best_genomes if best_genomes is not None else self.genomes
                            else:
                                print('Early stopping triggered')
                                break

                elif self.global_step % eval_freq == 0:
                    n = int(self.pop_size * 0.1)
                    avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return, \
                        avg_constraint_violations, front = \
                        multi_policy_evaluation_evorl(self, eval_envs, w=self.genomes[:n],
                                                      rep=num_eval_episodes_for_front)

                    self.report(
                        avg_scalarized_return,
                        avg_scalarized_discounted_return,
                        avg_vec_return,
                        avg_disc_vec_return,
                        avg_constraint_violations,
                        log_name='eval',
                    )
                    wandb.log({'num_episodes': self.num_episodes, 'training_time': round(total_training_time)})

        if not self.tune:
            self.global_step += 1

            best_algorithm, best_genomes = self.early_stopper.load_best_model()
            self.pop = best_algorithm if best_algorithm is not None else self.pop
            self.genomes = best_genomes if best_genomes is not None else self.genomes

            if eval_envs is not None and self.log:
                # Evaluation
                avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return, \
                    avg_constraint_violations, front = \
                    multi_policy_evaluation_evorl(self, eval_envs, w=self.genomes, rep=num_eval_episodes_for_front)

                self.report(
                    avg_scalarized_return,
                    avg_scalarized_discounted_return,
                    avg_vec_return,
                    avg_disc_vec_return,
                    avg_constraint_violations,
                    log_name='eval',
                )
                log_all_multi_policy_metrics(
                    current_front=front,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    ref_front=known_pareto_front,
                )
            self.save(sub_folder=sub_folder, filename=save_file_name)

        self.close_wandb()

    def get_network(self, genome: any):
        if self.algorithm == 'FF_NEAT':
            network = neat.nn.FeedForwardNetwork.create(genome, self.config)
        else:
            network = neat.nn.RecurrentNetwork.create(genome, self.config)
        return network

    def eval(self, obs: np.ndarray, network: any) -> Union[int, np.ndarray]:
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        actions = network.activate(obs)
        return actions

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
        network = self.get_network(genome)

        # Compute fitness values for each environment in the batch
        fitness_values = np.zeros((self.batch_size, self.reward_dim))
        for i, env in enumerate(self.envs_batch):
            fitness_values[i] = run_episode(env, network, self)

        # Aggregate fitness values (consider replacing with np.mean for average fitness)
        fitness_values = np.sum(fitness_values, axis=0) * (-1)

        # Return NSGA2Fitness object with aggregated fitness values
        return neat.nsga2.NSGA2Fitness(*fitness_values)

    def get_config(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'config_file': self.config_file,
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

    def modify_config_file(self, file_path) -> None:
        obs_dim = self.observation_dim
        output_dim = self.action_dim

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Loop through lines and modify the necessary lines
        for i, line in enumerate(lines):
            if line.strip().startswith('num_inputs'):
                lines[i] = f'num_inputs = {obs_dim}\n'
            elif line.strip().startswith('num_outputs'):
                lines[i] = f'num_outputs  = {output_dim}\n'
            elif line.strip().startswith('pop_size'):
                lines[i] = f'pop_size = {self.pop_size}\n'
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

    def save(self, sub_folder: str = None, filename: str = None):
        save_dir = f'{self.save_dir}/{sub_folder}'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}/{filename}', 'wb') as f:
            pickle.dump({
                'pop': self.pop,
                'config': self.config,
                'genomes': self.genomes
            }, f)

    def load(self, sub_folder: str = None, filename: str = None):
        save_dir = f'{self.save_dir}/{sub_folder}'
        if not os.path.exists(f'{save_dir}/{filename}'):
            raise ValueError(f'No saved model found at {save_dir}/{filename}')

        with open(f'{save_dir}/{filename}', 'rb') as f:
            checkpoint = pickle.load(f)
        self.pop = checkpoint['pop']
        self.config = checkpoint['config']
        self.genomes = checkpoint['genomes']
        print(self.genomes)

    def get_weights(self):
        return [self.get_network(genome) for genome in self.genomes]

    def update(self) -> None:
        pass
