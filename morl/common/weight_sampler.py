import time

import torch as th
import numpy as np
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from pymoo.operators.sampling.lhs import LHS
from morl.common.weights import random_weights, equally_spaced_weights
import random


class UniformWeightSampler:
    def __init__(self, reward_dim, seed):
        self.train_weights = equally_spaced_weights(reward_dim, n=int(1e8), seed=seed)

    def sample(self, n_sample):
        return th.FloatTensor(random.choices(self.train_weights, k=n_sample))


class RandomWeightSampler:
    def __init__(self, reward_dim, seed):
        self.train_weights = random_weights(reward_dim, dist='dirichlet', n=int(1e8), seed=seed)

    def sample(self, n_sample):
        return th.FloatTensor(random.choices(self.train_weights, k=n_sample))


class WeightProblem(Problem):
    def __init__(self, fitness_function, n_var: int, n_obj: int, low: float = 0, up: float = 1):
        self.fitness_function = fitness_function
        xl = [low] * n_var
        xu = [up] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        start_time = time.time()
        f = [self.fitness_function(params) for params in x]
        out['F'] = np.array(f)
        print('Execution time: ', time.time() - start_time)


class NSGAWeightSampler:
    def __init__(self, reward_dim: int, seed: int, agent: any, envs: any, batch_size: int = 4,
                 ga_iterations: int = 50, pop_size: int = 100, mut_prob: float = 0.2, mut_eta: float = 20,
                 sbx_prob: float = 0.9, sbx_eta: float = 15):
        self.reward_dim = reward_dim
        self.seed = seed
        self.agent = agent
        self.envs = envs
        self.batch_size = batch_size
        self.ga_iterations = ga_iterations
        self.rnd = random.Random(seed)

        mutation_operator = PM(prob=mut_prob, eta=mut_eta, vtype=float)
        crossover_operator = SBX(prob=sbx_prob, eta=sbx_eta, vtype=float)
        sampling_operator = LHS()

        self.algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling_operator,
            crossover=crossover_operator,
            mutation=mutation_operator
        )

        self.problem = WeightProblem(fitness_function=self.fitness, n_var=self.reward_dim, n_obj=self.reward_dim)

        self.res = minimize(self.problem, self.algorithm, ('n_gen', 1), save_history=False,
                            seed=self.seed)
        self.algorithm.pop = self.res.pop

    def update(self):
        self.res = minimize(self.problem, self.algorithm, ('n_gen', self.ga_iterations), save_history=False,
                            seed=self.seed)
        self.algorithm.pop = self.res.pop

    def fitness(self, params):
        envs = self.rnd.sample(self.envs, self.batch_size)
        fitness_values = np.zeros((self.batch_size, self.reward_dim))
        for i, env in enumerate(envs):
            fitness_values[i] = self.run_episode(env, params, self.agent)
        fitness_values = np.sum(fitness_values, axis=0)
        return fitness_values * -1

    def run_episode(self, env, w, agent):
        obs, episode_done = env.reset()[0], False
        reward = np.zeros(env.unwrapped.reward_dim)
        while not episode_done:
            obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
            episode_done = terminated or truncated
            reward += r
        return reward

    def get_pop(self):
        return self.res.pop.get('X')

    def sample(self, n_sample, noise_std=0.1):
        sampled_weights = random.choices(self.res.pop.get('X'), k=n_sample)
        noise = np.random.normal(0, noise_std, size=np.array(sampled_weights).shape)
        noisy_sampled_weights = np.array(sampled_weights) + noise
        noisy_sampled_weights = np.clip(noisy_sampled_weights, 0, 1)
        return th.FloatTensor(noisy_sampled_weights)
        # return th.FloatTensor(random.choices(self.res.pop.get('X'), k=n_sample))


class ExperienceBasedWeightSampler:
    def __init__(self, rwd_dim, smoothing_factor=0.1):  # history_len=100,
        self.rwd_dim = rwd_dim
        self.smoothing_factor = smoothing_factor
        # Initialize EMA performance metrics
        self.ema_performance = th.zeros(rwd_dim)

        # Initialize historical performance record
        # self.performance_history = th.zeros((history_len, rwd_dim))
        # self.history_idx = 0

    def update_performance(self, new_rewards):
        # Update the performance history with new rewards
        # self.performance_history[self.history_idx % len(self.performance_history)] = new_rewards
        # self.history_idx += 1
        if not isinstance(new_rewards, th.Tensor):
            new_rewards = th.tensor(new_rewards, dtype=th.float32)

        # Apply Exponential Moving Average
        self.ema_performance = self.smoothing_factor * new_rewards + \
                               (1 - self.smoothing_factor) * self.ema_performance

    def sample(self, n_sample):
        # performance = self.performance_history.mean(dim=0)
        performance = self.ema_performance
        if self.rwd_dim == 3:
            # Calculate emphasis based on historical performance (e.g., inverse of average rewards)
            emphasis = 1.0 / (performance + 1e-6)  # Adding a small constant to avoid division by zero
            emphasis = emphasis / emphasis.sum()  # Normalize

        else:
            # Handle constraint dimension differently
            constraint_emphasis = 1.0 / (performance[0] + 1e-6)  # Assuming more negative is worse
            other_emphasis = 1.0 / (performance[1:] + 1e-6)

            # Combine and normalize
            combined_emphasis = th.cat((constraint_emphasis.unsqueeze(0), other_emphasis))
            emphasis = combined_emphasis / combined_emphasis.sum()
        # Sample weights with emphasis on underperforming objectives

        weights = th.rand(n_sample, self.rwd_dim) * emphasis
        normalized_weights = weights / th.norm(weights, dim=1, keepdim=True, p=1)
        # normalized_weights = weights / weights.sum(dim=1, keepdim=True)
        # print(f'w: {normalized_weights}; sum: {normalized_weights.sum()}, emphasis: {emphasis}')
        return normalized_weights.float()


class WeightSamplerAngle:
    """Sample weight vectors from normal distribution."""

    def __init__(self, rwd_dim, angle, w=None):
        """Initialize the weight sampler."""
        self.rwd_dim = rwd_dim
        self.angle = angle
        if w is None:
            w = th.ones(rwd_dim)
        w = w / th.norm(w)
        self.w = w

    def sample(self, n_sample):
        """Sample n_sample weight vectors from normal distribution."""
        s = th.normal(th.zeros(n_sample, self.rwd_dim))

        # remove fluctuation on dir w
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)

        # normalize it
        s = s / th.norm(s, dim=1, keepdim=True)

        # sample angle
        s_angle = th.rand(n_sample, 1) * self.angle

        # compute shifted vector from w
        w_sample = th.tan(s_angle) * s + self.w.view(1, -1)

        w_sample = w_sample / th.norm(w_sample, dim=1, keepdim=True, p=1)

        weight = w_sample.float()
        return weight
