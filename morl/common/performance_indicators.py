"""Performance indicators for multi-objective RL algorithms.

We mostly rely on pymoo for the computation of axiomatic indicators (HV and IGD), but some are customly made.
"""
from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def epsilon_metric(front_p: List[npt.ArrayLike], front_q: List[npt.ArrayLike], n_objectives: int) -> float:
    epsilon = float('-inf')

    for point_q in front_q:
        epsilon_for_q = float('inf')
        for point_p in front_p:
            scaling_factor_for_pair = max(
                point_q[obj] / point_p[obj] if point_p[obj] != 0 else float('inf') for obj in range(n_objectives))
            epsilon_for_q = min(epsilon_for_q, scaling_factor_for_pair)
        epsilon = max(epsilon, epsilon_for_q)

    return epsilon


def sort_front_lexicographic(front: List[np.ndarray], maximization: bool = True):
    """
    Sort solutions based on their values in the objective space.
    Assumes each solution is an array of objective values.
    """
    # Sort solutions based on their objective values
    # Assuming a 2D or 3D objective space, sort primarily by the first objective,
    # then second, and so on.
    if maximization:
        sorted_front = sorted(front, key=lambda x: tuple(-y for y in x))
    else:
        sorted_front = sorted(front, key=lambda x: tuple(x))
    return sorted_front


# def spread(front: List[np.ndarray], reference_point: np.ndarray):
#     if len(front) < 2:
#         return 0.0
#     # Sort solutions
#     front = sort_solutions(front, reference_point)
#     # front = sort_solutions_by_objectives(front)
#     # print(front)
#     # Calculate distances between consecutive solutions
#     distances = [np.linalg.norm(front[i] - front[i - 1]) for i in range(1, len(front))]
#
#     # Calculate the average of these distances
#     d_mean = np.mean(distances)
#     # Include the distance from the boundary solutions to the extremes of the obtained front
#     d_f = np.linalg.norm(front[0] - front[1])  # distance from first solution to second solution
#     d_l = np.linalg.norm(front[-1] - front[-2])  # distance from last solution to second-to-last solution
#
#     # Calculate the Spread metric
#     # spread_value = sum(abs(d - d_mean) for d in distances) / sum(distances)
#     # spread_value = (sum(abs(d - d_mean) for d in distances) + abs(d_f - d_mean) + abs(d_l - d_mean)) / (
#     #            sum(distances) + d_f + d_l)
#     spread_value = (d_f + d_l + sum(abs(d - d_mean) for d in distances)) / (d_f + d_l + len(front) - 1 * d_mean)
#     return spread_value


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def sort_front_by_euclidean_distance(front: List[np.ndarray], reference_point: np.ndarray):
    # Calculate a composite index for each solution
    composite_indices = [euclidean_distance(point, reference_point) for point in front]
    # Sort solutions based on the composite index
    sorted_front = [sol for _, sol in sorted(zip(composite_indices, front))]
    return sorted_front


def calculate_distances_to_extremes(front, extreme_points):
    """Calculate the distances from each extreme point to the nearest solution."""
    distances = []
    for extreme in extreme_points:
        min_distance = min(euclidean_distance(extreme, point) for point in front)
        distances.append(min_distance)
    return distances


def find_extreme_points(front, num_objectives):
    extreme_points = []
    for i in range(num_objectives):
        # Find the solution with the maximum value for objective i
        extreme_point = max(front, key=lambda s: s[i])
        extreme_points.append(extreme_point)
    return extreme_points


def calculate_consecutive_distances(front):
    """Calculate the Euclidean distances between consecutive solutions."""
    distances = [euclidean_distance(front[i], front[i + 1]) for i in range(len(front) - 1)]
    return distances


def spread(front, reference_point: np.ndarray):
    """Calculate the generalized spread metric."""
    if len(front) < 2:
        return 0.0

    # Sort solutions based on their objectives (assuming solutions is a list of numpy arrays)
    # solutions = sort_solutions_lexicographic(solutions)
    front = sort_front_by_euclidean_distance(front, reference_point)
    # Calculate distances to extreme points
    extreme_points = find_extreme_points(front, len(front[0]))
    distances_to_extremes = calculate_distances_to_extremes(front, extreme_points)

    # Calculate distances between consecutive solutions
    consecutive_distances = calculate_consecutive_distances(front)
    d_mean = np.mean(consecutive_distances)

    # Compute the spread value
    numerator = sum(abs(d - d_mean) for d in consecutive_distances) + sum(distances_to_extremes)
    denominator = sum(consecutive_distances) + sum(distances_to_extremes)
    spread_value = numerator / denominator

    return spread_value


def spacing(front: List[np.ndarray], reference_point: np.ndarray):
    if len(front) < 2:
        return 0.0

    # front = sort_solutions(front, reference_point)
    front = sort_front_by_euclidean_distance(front, reference_point)
    # Calculate distances between consecutive solutions
    distances = [np.linalg.norm(front[i] - front[i - 1]) for i in range(1, len(front))]

    # Calculate the average of these distances
    d_mean = np.mean(distances)
    # Calculate the Spacing metric
    spacing_value = np.sqrt(np.mean([(d - d_mean) ** 2 for d in distances]))

    return spacing_value


def r2_indicator(front: List[np.ndarray], weights: List[np.ndarray], utopian_point: np.ndarray) -> float:
    """
    R2 indicator implementation based on Definition 4.

    Args:
        solutions: List of solution vectors (Pareto front approximation).
        weights: List of weight vectors (uniformly distributed over the weight space).
        utopian_point: The utopian point (ideal objective values).

    Returns:
        float: R2 indicator value.
    """
    if utopian_point is None:
        m = len(front[0])
        utopian_point = np.ones(m)

    def utility_function(point, weight, utopian_point):
        return max(weight[j] * abs(utopian_point[j] - point[j]) for j in range(len(point)))

    r2_value = (1 / len(weights)) * sum(
        min(utility_function(point, weight, utopian_point) for point in front) for weight in weights)
    return r2_value


def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """Inverted generational distance metric. Requires to know the optimal front.

    Args:
        known_front: known pareto front for the problem
        current_estimate: current pareto front

    Return:
        a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def maximum_utility_loss(
        front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in
                      range(len(max_scalarized_values))]
    return np.max(utility_losses)
