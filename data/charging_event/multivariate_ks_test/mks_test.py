# https://github.com/o-laurent/multivariate-ks-test/blob/main/mks_test/utils.py
from typing import Callable, Optional

import numpy as np


def mecdf(x_val: np.ndarray, t: np.ndarray) -> float:
    """Computes the multivariate empirical cdf of x_val at t.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        t: A numpy array of shape (num_samples_t, dim) representing the point at which to evaluate
            the cdf.

    Returns:
        The multivariate empirical cdf of x_val at t.
    """
    lower = (x_val <= t) * 1.0
    return np.mean(np.prod(lower, axis=1))


def ks_2samp(
        x_val: np.ndarray,
        y_val: np.ndarray,
        alpha: float,
        asymptotic: bool = False,
        verbose: bool = False,
) -> bool:
    """Performs a multivariate two-sample extension of the Kolmogorov-Smirnov test.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the first sample.
        y_val: A numpy array of shape (num_samples_y, dim) representing the second sample.
        alpha: The significance level.
        asymptotic: Whether to use the asymptotic approximation or not.
        verbose: Whether to print the test statistic and the critical value.

    Returns:
        A boolean indicating whether the null hypothesis is rejected.
    """
    print('Performs a multivariate two-sample extension of the Kolmogorov-Smirnov test.')
    num_samples_x, dim = x_val.shape
    num_samples_y, num_feats_y = y_val.shape

    if dim != num_feats_y:
        raise ValueError("The two samples do not have the same number of features.")

    z = np.zeros((num_samples_x, dim, dim))
    for h in range(dim):
        ind = np.argsort(x_val[:, h])[::-1]
        temp = np.take(x_val, ind, axis=0)
        z[:, :, h] = temp
        for i in range(dim):
            for j in range(num_samples_x - 1, -1, -1):
                if j == num_samples_x - 1:
                    runmax = temp[num_samples_x - 1, i]
                else:
                    runmax = max(runmax, temp[j, i])
                z[j, i, h] = runmax

    diff = np.zeros((num_samples_x, dim))
    for h in range(dim):
        for i in range(num_samples_x):
            val = np.abs(mecdf(x_val, z[i, :, h]) - mecdf(y_val, z[i, :, h])) * (
                    round(num_samples_x * mecdf(x_val, z[i, :, h])) == num_samples_x - i
            )
            diff[i, h] = val
            if h == 0:
                diff[i, h] = max(
                    diff[i, h],
                    np.abs(mecdf(x_val, x_val[i, :]) - mecdf(y_val, x_val[i, :])),
                )
    KS = np.max(diff)

    if asymptotic:
        KS_critical_val = np.sqrt(-np.log(alpha / (4 * dim)) * (0.5 / num_samples_x)) + np.sqrt(
            (-1) * np.log(alpha / (4 * dim)) * (0.5 / num_samples_y)
        )
    else:
        KS_critical_val = np.sqrt(
            -np.log(alpha / (2 * (num_samples_x + 1) * dim)) * (0.5 / num_samples_x)
        ) + np.sqrt((-1) * np.log(alpha / (2 * (num_samples_y + 1) * dim)) * (0.5 / num_samples_y))

    if verbose:
        print("test statistic: ", KS)
        print("test statistic critical value: ", KS_critical_val)
    return KS > KS_critical_val


def ks_1samp(
        x_val: np.ndarray,
        f_y: Callable,
        alpha: float,
        asymptotic: bool = False,
        verbose: bool = False,
) -> bool:
    """Performs a multivariate one-sample extension of the Kolmogorov-Smirnov test.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        f_y: The cdf of the distribution to test against.
        alpha: The significance level.
        asymptotic: Whether to use the asymptotic approximation or not.
        verbose: Whether to print the test statistic and the critical value.

    Returns:
        A boolean indicating whether the null hypothesis is rejected.
    """
    print('Performs a multivariate one-sample extension of the Kolmogorov-Smirnov test')
    num_samples_x, dim = x_val.shape

    z = np.zeros((num_samples_x, dim, dim))
    for h in range(dim):
        ind = np.argsort(x_val[:, h])[::-1]
        temp = np.take(x_val, ind, axis=0)
        z[:, :, h] = temp
        for i in range(dim):
            for j in range(num_samples_x - 1, -1, -1):
                if j == num_samples_x - 1:
                    runmax = temp[num_samples_x - 1, i]
                else:
                    runmax = max(runmax, temp[j, i])
                z[j, i, h] = runmax

    diff = np.zeros((num_samples_x, dim))
    for h in range(dim):
        for i in range(num_samples_x):
            val = np.abs(mecdf(x_val, z[i, :, h]) - f_y(z[i, :, h])) * (
                    round(num_samples_x * mecdf(x_val, z[i, :, h])) == num_samples_x - i
            )
            diff[i, h] = val
            if h == 0:
                diff[i, h] = max(
                    diff[i, h],
                    np.abs(mecdf(x_val, x_val[i, :]) - f_y(x_val[i, :])),
                )
    KS = np.max(diff)
    if asymptotic:
        KS_critical_val = np.sqrt(-np.log(alpha / (2 * dim)) * (0.5 / num_samples_x))
    else:
        KS_critical_val = np.sqrt(
            -np.log(alpha / (2 * (num_samples_x + 1) * dim)) * (0.5 / num_samples_x)
        )

    if verbose:
        print("test statistic: ", KS)
        print("test statistic critical value: ", KS_critical_val)
    return KS > KS_critical_val


def mkstest(
        x_val: np.ndarray,
        y_val: Optional[np.ndarray] = None,
        y_cdf: Optional[Callable] = None,
        alpha: float = 0.05,
        verbose: bool = False,
) -> bool:
    """Extended Kolmogorov-Smirnov test for the one and two-sample cases.

    Args:
        x_val (array_like): first data sample.
        y_val (array_like): second data sample.
        y_cdf (callable): mcdf of the distribution to test against in the one-sample case.
        alpha (float): significance level.
        verbose (bool): whether to print the test statistic and the critical value.

    Returns:
        bool: True if the null hypothesis is rejected, False otherwise.
    """
    if not isinstance(alpha, float):
        raise ValueError("alpha must be float")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")
    if y_val is None and y_cdf is None:
        raise ValueError("y_val and y_cdf cannot both be None")

    if y_cdf:
        return ks_1samp(x_val, y_cdf, alpha=alpha, verbose=verbose)
    return ks_2samp(x_val, y_val, alpha=alpha, verbose=verbose)
