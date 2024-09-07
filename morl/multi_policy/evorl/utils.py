import time
import torch.cuda
import numpy as np
from typing import Any
from copy import deepcopy
from joblib import Parallel, delayed
from pymoo.core.problem import Problem


class EarlyStopping:
    def __init__(self, patience: int = 5, restarts: int = 3, min_delta: float = 0):
        self.patience = patience
        self.restarts = restarts
        self.min_delta = min_delta
        self.best_metric = float('-inf')
        self.best_algorithm = None
        self.best_genomes = None
        self.patience_counter = 0
        self.restarts_counter = 0

    def __call__(self, metric, algorithm, genomes):
        if metric - self.min_delta > self.best_metric:
            self.best_metric = metric
            self.best_algorithm = deepcopy(algorithm)
            self.best_genomes = deepcopy(genomes)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.restarts_counter += 1
                return True  # Indicate early stopping
        return False

    def do_restart(self):
        if self.restarts_counter <= self.restarts:
            self.patience_counter = 0
            return True
        return False

    def load_best_model(self):
        return self.best_algorithm, self.best_genomes


class NeuralNetworkProblem(Problem):
    def __init__(self, fitness_function, n_var: int, n_obj: int, n_jobs: int = -1, low: float = -3, up: float = 3):
        self.fitness_function = fitness_function
        self.n_jobs = n_jobs
        # for relu activations the weights ranges should [-2, 2] or [-3, 3]
        xl = [low] * n_var
        xu = [up] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        start_time = time.time()
        try:
            # Ensure tensors are of dtype float and moved to the appropriate device
            x = [torch.tensor(param, dtype=torch.float, device='cpu') for param in x]

            with Parallel(n_jobs=self.n_jobs, backend='loky') as parallel:
                f = parallel(delayed(self.fitness_function)(params) for params in x)
            out['F'] = np.array(f)

        except Exception as e:
            raise Exception(f'An error occurred: {e}')

        finally:
            print('Execution time: ', time.time() - start_time)


def run_episode(env: Any, network: Any, agent: Any) -> np.ndarray:
    obs, episode_done = env.reset()[0], False
    reward = np.zeros(env.unwrapped.reward_dim)
    while not episode_done:
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, network))
        episode_done = terminated or truncated
        reward += r
    return reward
