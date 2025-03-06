import pickle
import numpy as np
import seaborn as sns

from hyppo.ksample import Energy, MMD

from work_charging_events import get_work_data
from charging_events_utils import create_real_samples, remove_outliers, filter_samples

sns.set()
np.random.seed(42)


def generate_samples(gmm_file: str, n_samples: int) -> np.ndarray:
    """
    Load a Gaussian Mixture Model from a file and generate samples from it.

    :param gmm_file: The filename of the saved Gaussian Mixture Model.
    :param n_samples: The number of samples to generate.
    :return: A tuple containing lists of generated arrival times, loading times, and energy amounts.
    """
    with open(f'models/{gmm_file}.pkl', 'rb') as f:
        gmm_model = pickle.load(f)
    generated_samples, _ = gmm_model.sample(n_samples)

    generated_samples = filter_samples(generated_samples)

    return generated_samples


def main():
    n_samples = 10000

    gmm_file = f'gmm_work_weekday_70'
    df = get_work_data()

    real_samples = create_real_samples(df, 100000)
    indices = np.random.randint(0, len(real_samples), n_samples)

    real_samples = real_samples[indices]
    real_samples = filter_samples(real_samples)
    Y = remove_outliers(real_samples)
    X = generate_samples(gmm_file, n_samples)

    # Hypothesis test using Energy Distance
    stats, p_value = Energy().test(X, Y, reps=1000)
    # stats, p_value = MMD(compute_kernel='rbf').test(X, Y, reps=1000)
    if p_value > 0.05:
        print('p-value is > 0.05, so we fail to reject the null hypothesis.')
        print('Interpretation: The test does NOT find a statistically significant difference.')
    else:
        print('p-value is <= 0.05, so we reject the null hypothesis.')
        print('Interpretation: The test DOES find a statistically significant difference between the distributions.')

    print('Statistic:', stats)
    print('p-value:', p_value)


if __name__ == '__main__':
    main()
