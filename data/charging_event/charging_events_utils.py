import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from simulation.constant import MAX_BATTERY_CAP, MIN_ENERGY, MIN_CHARGING_TIME, MAX_CHARGING_TIME, GMM_SEED


def time_to_minutes(timestamp: pd.Timestamp) -> float:
    """
    Converts given time to minutes.

    :param timestamp: The time to be converted to minutes.
    :return: Time in minutes.
    """
    hour, minutes, second = timestamp.hour, timestamp.minute, timestamp.second
    minutes = int(hour) * 60 + int(minutes)
    if int(second) > 30:
        minutes = minutes + 1
    return minutes


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame to generate a new DataFrame with relevant features.

    :param df: The original DataFrame to be processed.
    :return: The processed DataFrame.
    """
    df = df[['arrival_time', 'departure_time', 'energy']].copy()  # Ensure that we have a copy of the DataFrame
    df.loc[:, 'charging_time'] = (df['departure_time'] - df[
        'arrival_time']).dt.total_seconds() / 60  # Loading time in minutes
    df.loc[:, 'weekday'] = ((pd.DatetimeIndex(df['arrival_time']).dayofweek) // 5).astype(int)
    df['arrival_time_minutes'] = df['arrival_time'].apply(time_to_minutes)
    # remove all charges where nothing in charged or duration is lower then 5 minutes
    df = df[
        (df['energy'] >= 0) & (df['energy'] <= MAX_BATTERY_CAP) & (df['charging_time'] >= MIN_CHARGING_TIME) & (
                df['charging_time'] <= MAX_CHARGING_TIME)]
    df = df.dropna()
    return df


def create_real_samples(df: pd.DataFrame, n_samples: int = 100000) -> np.ndarray:
    """
    Create real samples of size n_samples based on weekday or weekend.

    :param df: DataFrame with all data.
    :param n_samples: Number of samples to create.
    :return: Array of samples.
    """
    df = df[df['weekday'] == 0]

    # xi = (ai ,di , ei ) in R3 where ai denotes the arrival time, di denotes the duration and ei is the total energy (in kWh) delivered
    x = df[['arrival_time_minutes', 'charging_time', 'energy']].values
    # draw 100 000 samples
    if x.shape[0] > n_samples:
        indices = np.random.choice(x.shape[0], size=int(n_samples), replace=False)
        x = x[indices]
    else:
        print(f'The array has fewer than {n_samples} rows!')
    return x


def remove_outliers(x: np.ndarray) -> np.ndarray:
    """
    Removes outliers from an array using IsolationForest.

    :param x: Array with possible outliers.
    :return: Array with outliers removed.
    """
    # define the model
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=GMM_SEED)  # change contamination parameter as needed

    # fit the model
    clf.fit(x)

    # get the prediction labels of the training data
    pred = clf.predict(x)  # -1 for outliers and 1 for inliers

    # filter outlier from data_array
    filtered_x = x[pred == 1]
    return filtered_x


def gmm_fitting(df: pd.DataFrame, weekday: bool, name: str) -> None:
    """
     Fits a Gaussian Mixture Model to the data and saves the model.

     :param df: DataFrame with all data.
     :param weekday: True if it's weekday, False if it's weekend.
     :param name: The name for the saved model.
     :return: None
     """
    x = create_real_samples(df)
    x = filter_samples(x)
    x = remove_outliers(x)
    best_bic = np.inf
    best_n_components = None

    for n_components in range(1, 70 + 1):
        gmm = GaussianMixture(n_components=n_components, init_params='kmeans', random_state=GMM_SEED)
        gmm.fit(x)
        bic = gmm.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_n_components = n_components

    print(f'The best model has {best_n_components} components.')

    path = 'models'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/gmm_{name}_weekday_{best_n_components}.pkl', 'wb') as f:
        pickle.dump(gmm, f)


def filter_samples(samples: np.ndarray) -> np.ndarray:
    samples = samples[(samples[:, 0] >= 0) & (samples[:, 0] <= 1440) &
                                (samples[:, 1] <= MAX_CHARGING_TIME) & (
                                        samples[:, 1] >= MIN_CHARGING_TIME) &
                                (samples[:, 2] >= (MIN_ENERGY/1000)) & (
                                        samples[:, 2] <= (MAX_BATTERY_CAP / 1000))]
    return samples