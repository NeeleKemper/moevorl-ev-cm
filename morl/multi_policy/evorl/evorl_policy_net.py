import os
import pickle
import random
import time
import warnings
import torch
import wandb
import numpy as np
import torch.nn as nn
from typing import List, Optional, Union

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS

from morl.common.evaluation import seed_everything
from morl.multi_policy.evorl.utils import run_episode, NeuralNetworkProblem, EarlyStopping
from morl.common.morl_algorithm import MOAgent, MOPolicy
from morl.common.evaluation import (
    log_all_multi_policy_metrics,
    multi_policy_evaluation_evorl
)
from envs.ev_charging.ev_charging import EVCharging
from envs.ev_charging_test.ev_charging_test import EVChargingTest

warnings.filterwarnings('ignore')


class LSTM(nn.Module):
    def __init__(self, obs_dim: int, output_dim: int, hidden_size: int = 8, num_layers: int = 3, dropout: float = 0,
                 bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=obs_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            # Ensure dropout is only applied if num_layers > 1
                            bidirectional=bidirectional)

        num_directions = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_size * num_directions, output_dim)

    def forward(self, obs):
        # obs should have dimensions: (batch_size, sequence_length, obs_dim)
        obs = obs.unsqueeze(0).unsqueeze(0)
        rnn_out, _ = self.lstm(obs)
        # Select the output of the last time step
        last_seq = rnn_out[:, -1, :]
        action = self.output_layer(last_seq)
        return torch.sigmoid(action)

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x: list):
        start = 0
        x_tensor = torch.FloatTensor(x)
        for p in self.parameters():
            end = start + p.numel()  # equivalent to np.prod(p.shape)
            p.data.copy_(x_tensor[start:end].view(p.shape))
            start = end


class RNN(nn.Module):
    def __init__(self, obs_dim: int, output_dim: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = False):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=obs_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          nonlinearity='tanh',
                          bias=True,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,  # Ensure dropout is only applied if num_layers > 1
                          bidirectional=bidirectional)

        num_directions = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_size * num_directions, output_dim)

    def forward(self, obs):
        # obs should have dimensions: (batch_size, sequence_length, obs_dim)
        obs = obs.unsqueeze(0).unsqueeze(0)
        rnn_out, _ = self.rnn(obs)
        # Select the output of the last time step
        last_seq = rnn_out[:, -1, :]
        action = self.output_layer(last_seq)
        return torch.sigmoid(action)

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x: list):
        start = 0
        x_tensor = torch.FloatTensor(x)
        for p in self.parameters():
            end = start + p.numel()  # equivalent to np.prod(p.shape)
            p.data.copy_(x_tensor[start:end].view(p.shape))
            start = end


class FFNeuralNetwork(nn.Module):
    def __init__(self, obs_dim: int, output_dim: int, net_arch: list = [256, 256]):
        super(FFNeuralNetwork, self).__init__()

        layers = [nn.Linear(obs_dim, net_arch[0]), nn.ReLU()]
        layers.extend([func for n in net_arch[:-1] for func in [nn.Linear(n, n), nn.ReLU()]])

        self.latent_pi = nn.Sequential(*layers)
        self.mean = nn.Linear(net_arch[-1], output_dim)

    def forward(self, obs):
        h = self.latent_pi(obs.float())
        action = self.mean(h)
        return torch.sigmoid(action)

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x: list):
        start = 0
        x_tensor = torch.FloatTensor(x)
        for p in self.parameters():
            end = start + p.numel()  # equivalent to np.prod(p.shape)
            p.data.copy_(x_tensor[start:end].view(p.shape))
            start = end


class EvoRLPolicyNet(MOAgent, MOPolicy):
    def __init__(self,
                 envs: List[EVCharging | EVChargingTest],
                 net_arch: List = [256, 256],
                 rnn_hidden_size: int = 8,
                 rnn_num_layers: int = 3,
                 rnn_dropout: float = 0,
                 rnn_bidirectional: bool = False,
                 pop_size: int = 100,
                 sbx_prob: float = 0.9,
                 sbx_eta: float = 10,
                 mut_prob: float = 0.9,
                 mut_eta: float = 10,
                 batch_size: int = 32,
                 env_iterations: int = 10,
                 algorithm: str = 'NSGA2',
                 network_type: str = 'FF',
                 project_name: str = 'MORL',
                 experiment_name: str = 'EvoRL_FF_NSGA2',
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

        if device == 'auto':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        MOAgent.__init__(self, envs[0], device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        self.discrete_actions = False
        self.envs = envs
        self.net_arch = net_arch
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.rnn_bidirectional = rnn_bidirectional
        self.pop_size = pop_size
        self.sbx_prob = sbx_prob
        self.sbx_eta = sbx_eta
        self.mut_prob = mut_prob
        self.mut_eta = mut_eta
        self.batch_size = batch_size
        self.env_iterations = env_iterations
        self.algorithm_name = algorithm
        self.network_type = network_type
        if network_type == 'RNN':
            policy_network = RNN(obs_dim=self.observation_dim,
                                 output_dim=self.action_dim,
                                 hidden_size=self.rnn_hidden_size,
                                 num_layers=self.rnn_num_layers,
                                 dropout=self.rnn_dropout,
                                 bidirectional=self.rnn_bidirectional).to(device)
        elif network_type == 'LSTM':
            policy_network = LSTM(obs_dim=self.observation_dim,
                                  output_dim=self.action_dim,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=self.rnn_num_layers,
                                  dropout=self.rnn_dropout,
                                  bidirectional=self.rnn_bidirectional).to(device)
        else:
            policy_network = FFNeuralNetwork(
                obs_dim=self.observation_dim,
                output_dim=self.action_dim,
                net_arch=net_arch).to(device)

        self.initial_mut_prob = mut_prob
        self.initial_sbx_prob = sbx_prob
        mutation_operator = PM(prob=mut_prob, eta=mut_eta, vtype=float)
        crossover_operator = SBX(prob=sbx_prob, eta=sbx_eta, vtype=float)

        sampling_operator = LHS()  # Get Latin Hypercube Sampling, FloatRandomSampling()

        if algorithm == 'NSGA2':
            self.algorithm = NSGA2(
                pop_size=pop_size,
                sampling=sampling_operator,
                crossover=crossover_operator,
                mutation=mutation_operator
            )
        else:  # self.algorithm == 'SPEA2'
            self.algorithm = SPEA2(
                pop_size=pop_size,
                sampling=sampling_operator,
                crossover=crossover_operator,
                mutation=mutation_operator
            )
        self.save_dir = f'models/{network_type.lower()}_{algorithm.lower()}'

        print(policy_network)
        print('#params', len(policy_network.get_params()))
        self.problem = NeuralNetworkProblem(fitness_function=self.fitness, n_var=len(policy_network.get_params()),
                                            n_obj=self.reward_dim, n_jobs=n_jobs)
        self.res = None
        self.genomes = None
        self.envs_batch = None
        self.early_stopper = EarlyStopping(patience=10, restarts=0, min_delta=0)

        self.log = log
        self.tune = tune
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

        torch.set_default_tensor_type('torch.FloatTensor')

    def get_config(self):
        if self.network_type == 'RNN' or self.network_type == 'LSTM':
            return {
                'env_id': self.env.unwrapped.spec.id,
                'algorithm': self.algorithm_name,
                'hidden_size': self.rnn_hidden_size,
                'num_layers': self.rnn_num_layers,
                'dropout': self.rnn_dropout,
                'bidirectional': self.rnn_bidirectional,
                'pop_size': self.pop_size,
                'sbx_prob': self.sbx_prob,
                'sbx_eta': self.sbx_eta,
                'mut_prob': self.mut_prob,
                'mut_eta': self.mut_eta,
                'batch_size': self.batch_size,
                'env_iterations': self.env_iterations,
                'seed': self.seed
            }
        else:
            return {
                'env_id': self.env.unwrapped.spec.id,
                'algorithm': self.algorithm_name,
                'net_arch': self.net_arch,
                'pop_size': self.pop_size,
                'sbx_prob': self.sbx_prob,
                'sbx_eta': self.sbx_eta,
                'mut_prob': self.mut_prob,
                'mut_eta': self.mut_eta,
                'batch_size': self.batch_size,
                'env_iterations': self.env_iterations,
                'seed': self.seed
            }

    def save(self, sub_folder: str = None, filename: str = None):
        save_dir = f'{self.save_dir}/{sub_folder}'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}/{filename}', 'wb') as f:
            pickle.dump({
                'algorithm': self.algorithm,
                'genomes': self.genomes
            }, f)

    def load(self, sub_folder: str = None, filename: str = None):
        save_dir = f'{self.save_dir}/{sub_folder}'
        if not os.path.exists(f'{save_dir}/{filename}'):
            raise ValueError(f'No saved model found at {save_dir}/{filename}')

        with open(f'{save_dir}/{filename}', 'rb') as f:
            checkpoint = pickle.load(f)
        self.algorithm = checkpoint['algorithm']
        self.genomes = checkpoint['genomes']

    def get_network(self, weights: any):
        if self.network_type == 'RNN':
            network = RNN(obs_dim=self.observation_dim,
                          output_dim=self.action_dim,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_num_layers,
                          dropout=self.rnn_dropout,
                          bidirectional=self.rnn_bidirectional).to(self.device)
        elif self.network_type == 'LSTM':
            network = LSTM(obs_dim=self.observation_dim,
                           output_dim=self.action_dim,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.rnn_num_layers,
                           dropout=self.rnn_dropout,
                           bidirectional=self.rnn_bidirectional).to(self.device)
        else:
            network = FFNeuralNetwork(
                obs_dim=self.observation_dim,
                output_dim=self.action_dim,
                net_arch=self.net_arch).to(self.device)
        network.set_params(weights)
        return network

    def eval(self, obs: np.ndarray, network: any) -> Union[int, np.ndarray]:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs).float().to(self.device)
        actions = network(obs)

        return actions.detach().cpu().numpy().flatten()

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
            save_file_name: str = 'EvoRL_FF'
    ):
        assert eval_freq % self.env_iterations == 0, \
            f'The variable eval_freq must be a multiple of env_iterations. Got {eval_freq} and {self.env_iterations}'

        if self.log:
            self.register_additional_config({'ref_point': ref_point.tolist(), 'known_front': known_pareto_front})

        # Optimization using NSGA2 in chunks
        generations = 0
        total_training_time = 0
        while generations < n_generations:
            # train loop

            self.envs_batch = self.rnd.sample(self.envs, self.batch_size)
            self.num_episodes += (self.batch_size * self.env_iterations)

            generation_start_time = time.time()
            self.res = minimize(self.problem, self.algorithm, ('n_gen', self.env_iterations), save_history=False,
                                seed=self.seed)
            generation_end_time = time.time()
            total_training_time += (generation_end_time - generation_start_time)

            self.algorithm.pop = self.res.pop
            generations += self.env_iterations
            self.global_step = generations

            self.genomes = self.res.pop.get('X')

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
            self.algorithm = best_algorithm if best_algorithm is not None else self.algorithm
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

    def fitness(self, params):
        # start_time = time.time()
        policy = self.get_network(params)
        fitness_values = np.zeros((self.batch_size, self.reward_dim))
        for i, env in enumerate(self.envs_batch):
            fitness_values[i] = run_episode(env, policy, self)
        fitness_values = np.sum(fitness_values, axis=0)
        return fitness_values * -1

    def get_weights(self):
        return [self.get_network(genome) for genome in self.genomes]

    def update(self) -> None:
        pass
