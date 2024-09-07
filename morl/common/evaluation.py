"""Utilities related to evaluation."""
import copy
import os
import random
from typing import List, Optional, Tuple
from typing import Any
import numpy as np
import torch as th
import wandb
from numpy import ndarray
from pymoo.util.ref_dirs import get_reference_directions

from morl.common.pareto import filter_pareto_dominated
from morl.common.performance_indicators import (
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
    sparsity,
)
from morl.common.weights import equally_spaced_weights


def run_episode(env: Any, agent: Any, w: Any) -> tuple:
    obs, episode_done = env.reset()[0], False
    vec_return, disc_vec_return = np.zeros(env.unwrapped.reward_dim), np.zeros(env.unwrapped.reward_dim)
    gamma_value = 1.0

    while not episode_done:
        if hasattr(agent, "gamma"):
            obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
            gamma_value *= agent.gamma
        else:
            action = agent.eval(obs, w)
            obs, r, terminated, truncated, info = env.step(action)
            gamma_value *= 0.99
        episode_done = terminated or truncated
        if env.reward_dim == 3:
            if r[0] > 0:
                r = r / 100
        else:
            r[1] = r[1] / 100
            r[2] = r[2] / 100
            r[3] = r[3] / 100
        vec_return += r
        disc_vec_return += gamma_value * r
    return vec_return, disc_vec_return


def eval_mo(
        agent,
        envs,
        w: Optional[np.ndarray] = None,
        scalarization=np.dot,
        render: bool = False,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Evaluates one episode of the agent in the environment.

    Args:
        agent: Agent
        env: MO-Gymnasium environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return, mean return
    """
    n = len(envs)
    m = len(w)

    vec_returns, disc_vec_returns = np.zeros((n, m)), np.zeros((n, m))

    for i, env in enumerate(envs):
        vec_return, disc_vec_return = run_episode(env, agent, w)
        vec_returns[i], disc_vec_returns[i] = vec_return, disc_vec_return

    count = np.sum(vec_returns[:, 0] < 0)
    constraint_violations = count / len(vec_returns)

    vec_returns = np.mean(vec_returns, axis=0)
    disc_vec_returns = np.mean(disc_vec_returns, axis=0)

    if w is None:
        scalarized_return = vec_returns
        scalarized_discounted_return = disc_vec_returns
    else:
        scalarized_return = scalarization(w, vec_returns)
        scalarized_discounted_return = scalarization(w, disc_vec_returns)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_returns,
        disc_vec_returns,
        constraint_violations,
    )


def eval_evorl(agent, envs, w: list):
    local_policy = agent.get_network(w)
    n = len(envs)
    reward_dim = envs[0].unwrapped.reward_dim
    vec_returns, disc_vec_returns = np.zeros((n, reward_dim)), np.zeros((n, reward_dim))

    for i, env in enumerate(envs):
        vec_return, disc_vec_return = run_episode(env, agent, local_policy)
        vec_returns[i], disc_vec_returns[i] = vec_return, disc_vec_return

    count = np.sum(vec_returns[:, 0] < 0)
    constraint_violations = count / len(vec_returns)

    vec_return = np.mean(vec_returns, axis=0)
    disc_vec_return = np.mean(disc_vec_returns, axis=0)

    scalarized_return = vec_return
    scalarized_discounted_return = disc_vec_return

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
        constraint_violations,
    )


def eval_mo_reward_conditioned(
        agent,
        env,
        scalarization=np.dot,
        w: Optional[np.ndarray] = None,
        render: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment. This makes the assumption that the agent is conditioned on the accrued reward i.e. for ESR agent.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        scalarization: scalarization function, taking weights and reward as parameters
        w: weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return
    """
    obs, _ = env.reset()
    done = False
    vec_return, disc_vec_return = np.zeros(env.reward_space.shape[0]), np.zeros(env.reward_space.shape[0])
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, disc_vec_return))
        done = terminated or truncated
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= agent.gamma
    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )


def multi_policy_evaluation(agent, eval_envs, w: list, rep: int = 5):
    evals = [
        policy_evaluation_mo(agent, eval_envs, ew, rep=rep) for ew in w]

    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
    avg_constraint_violations = np.mean([eval[4] for eval in evals])
    front = [eval[2] for eval in evals]
    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
        avg_constraint_violations,
        front
    )


def multi_policy_evaluation_evorl(agent, eval_envs, w: list, rep: int = 5):
    evals = [
        policy_evaluation_evorl(agent, eval_envs, ew, rep=rep) for ew in w]

    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
    avg_constraint_violations = np.mean([eval[4] for eval in evals])
    front = [eval[2] for eval in evals]
    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
        avg_constraint_violations,
        front
    )


def policy_evaluation_evorl(agent, env, w: list, rep: int = 5) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        w (np.ndarray): network weights
        rep (int, optional): Number of episodes for averaging. Defaults to 5.

    Returns:
        (float, float, np.ndarray, np.ndarray, np.ndarray, float): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return, Avg mean reward
    """
    evals = [eval_evorl(agent, env, w) for _ in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
    avg_constraint_violations = np.mean([eval[4] for eval in evals])
    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
        avg_constraint_violations
    )


def policy_evaluation_mo(agent, env, w: np.ndarray, rep: int = 5) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        w (np.ndarray): Weight vector
        rep (int, optional): Number of episodes for averaging. Defaults to 5.

    Returns:
        (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return, Avg mean reward
    """
    evals = [eval_mo(agent, env, w) for _ in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
    avg_constraint_violations = np.mean([eval[4] for eval in evals])
    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
        avg_constraint_violations,
    )


def log_all_multi_policy_metrics(
        current_front: List[np.ndarray],
        hv_ref_point: np.ndarray,
        reward_dim: int,
        global_step: int,
        n_sample_weights: int = 50,
        ref_front: Optional[List[np.ndarray]] = None,
):
    """Logs all metrics for multi-policy training.

    Logged metrics:
    - hypervolume
    - sparsity
    - expected utility metric (EUM)
    If a reference front is provided, also logs:
    - Inverted generational distance (IGD)
    - Maximum utility loss (MUL)

    Args:
        current_front (List) : current Pareto front approximation, computed in an evaluation step
        hv_ref_point: reference point for hypervolume computation
        reward_dim: number of objectives
        global_step: global step for logging
        n_sample_weights: number of weights to sample for EUM and MUL computation
        ref_front: reference front, if known
    """
    filtered_front = list(filter_pareto_dominated(current_front))
    hv = hypervolume(hv_ref_point, filtered_front)
    sp = sparsity(filtered_front)
    eum = expected_utility(filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights))

    wandb.log(
        {
            "eval/hypervolume": hv,
            "eval/sparsity": sp,
            "eval/eum": eum,
            "global_step": global_step,
        },
        commit=False,
    )
    front = wandb.Table(
        columns=[f"objective_{i}" for i in range(1, reward_dim + 1)],
        data=[p.tolist() for p in filtered_front],
    )
    wandb.log({"eval/front": front})

    # If PF is known, log the additional metrics
    if ref_front is not None:
        generational_distance = igd(known_front=ref_front, current_estimate=filtered_front)
        mul = maximum_utility_loss(
            front=filtered_front,
            reference_set=ref_front,
            weights_set=get_reference_directions("energy", reward_dim, n_sample_weights).astype(np.float32),
        )
        wandb.log({"eval/igd": generational_distance, "eval/mul": mul})


def seed_everything(seed: int):
    """Set random seeds for reproducibility.

    This function should be called only once per python process, preferably at the beginning of the main script.
    It has global effects on the random state of the python process, so it should be used with care.

    Args:
        seed: random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


def log_episode_info(
        info: dict,
        scalarization,
        weights: Optional[np.ndarray],
        global_timestep: int,
        id: Optional[int] = None,
        verbose: bool = True,
):
    """Logs information of the last episode from the info dict (automatically filled by the RecordStatisticsWrapper).

    Args:
        info: info dictionary containing the episode statistics
        scalarization: scalarization function
        weights: weights to be used in the scalarization
        global_timestep: global timestep
        id: agent's id
        verbose: whether to print the episode info
    """
    episode_ts = info["l"]
    episode_time = info["t"]
    episode_return = info["r"]
    disc_episode_return = info["dr"]
    if weights is None:
        scal_return = scalarization(episode_return)
        disc_scal_return = scalarization(disc_episode_return)
    else:
        scal_return = scalarization(episode_return, weights)
        disc_scal_return = scalarization(disc_episode_return, weights)

    if verbose:
        print("Episode infos:")
        print(f"Steps: {episode_ts}, Time: {episode_time}")
        print(f"Total Reward: {episode_return}, Discounted: {disc_episode_return}")
        print(f"Scalarized Reward: {scal_return}, Discounted: {disc_scal_return}")

    if id is not None:
        idstr = "_" + str(id)
    else:
        idstr = ""
    wandb.log(
        {
            f"charts{idstr}/timesteps_per_episode": episode_ts,
            f"charts{idstr}/episode_time": episode_time,
            f"metrics{idstr}/scalarized_episode_return": scal_return,
            f"metrics{idstr}/discounted_scalarized_episode_return": disc_scal_return,
            "global_step": global_timestep,
        },
        commit=False,
    )

    for i in range(episode_return.shape[0]):
        wandb.log(
            {
                f"metrics{idstr}/episode_return_obj_{i}": episode_return[i],
                f"metrics{idstr}/disc_episode_return_obj_{i}": disc_episode_return[i],
            },
        )
