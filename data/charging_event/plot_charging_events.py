import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from scipy.interpolate import griddata
from matplotlib import cm
from work_charging_events import get_work_data
from charging_events_utils import create_real_samples, remove_outliers, filter_samples
from simulation.constant import MAX_BATTERY_CAP, MAX_CHARGING_TIME

sns.set()
np.random.seed(42)


def generate_samples(gmm_file: str, n_samples: int) -> Tuple[list, list, list]:
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

    generated_arrival_time = generated_samples[:, 0].tolist()
    generated_loading_time = generated_samples[:, 1].tolist()
    generated_energy = generated_samples[:, 2].tolist()
    return generated_arrival_time, generated_loading_time, generated_energy


def plot_samples(data: list, max_loading_time: float, max_energy: float, main_title: str, save_as: str) -> None:
    """
    Plot histograms of arrival time, loading time and energy for real and generated data.

    :param data: A list containing lists of arrival times, loading times, and energy amounts for real and generated data.
    :param max_loading_time: The maximum value for loading time. Used for setting x-axis limits.
    :param max_energy: The maximum value for energy. Used for setting x-axis limits.
    :param main_title: The main title for the plot.
    :param save_as: The filename to save the plot as.
    """
    sns.set_style("whitegrid")
    font_size = 18

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, sharey='row', figsize=(12, 10))  # figsize=(8, 8)

    # Define the data and labels for the plots

    labels = ['Arrival\nTime', 'Charging\nTime', 'Energy\nAmount']

    bins = [int(MAX_CHARGING_TIME / 24 / 2), int(MAX_CHARGING_TIME / 24 / 2), int(MAX_BATTERY_CAP / 2000)]

    # Define the colors and titles for the plots
    colors = ['b', 'r']
    titles = ['ACN Data', 'GMM Simulated Data']

    # Define the hour ticks and labels
    hours = list(range(0, 25, 3))  # Hours for ticks, here ticks are set for every 3 hours
    hour_labels = [str(hour) for hour in hours]  # Convert hours to string labels

    # Plot histograms and set labels
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.hist(data[i][j], bins=bins[i], color=colors[j], alpha=0.7, label=labels[i])
            ax.set_xlim(0, int(max_loading_time) if i < 2 else int(max_energy))
            ax.set_xlabel('Time [h]' if i < 2 else 'Energy [kWh]', fontweight='bold', fontsize=font_size)
            if j == 0:
                ax.set_ylabel('Count', fontweight='bold', fontsize=font_size)
            if i == 0:
                ax.set_title(titles[j], fontweight='bold', fontsize=font_size)
            # Set x-ticks and labels only for the first two rows of subplots
            if i < 2:
                ax.set_xticks([hour * 60 for hour in hours])  # Convert hours to minutes for ticks
                ax.set_xticklabels(hour_labels, fontsize=font_size)  # Set the x-axis labels as hour labels
            # Set the tick label font size for both x and y ticks
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)
            ax.legend(fontsize=font_size)


    # Set main title
    # fig.suptitle(main_title, fontsize=14, y=0.98, fontweight='bold')

    # Display the plots
    plt.tight_layout()

    path = 'plots'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{save_as}_paper.png')
    plt.show()


def plot_relations(data: list, max_loading_time: float, max_energy: float, main_title: str, save_as: str) -> None:
    """
    Plot scatter plots to show relations between arrival time, loading time, and energy for real and generated data.

    :param data: A list containing lists of arrival times, loading times, and energy amounts for real and generated data.
    :param max_loading_time: The maximum value for loading time. Used for setting x-axis limits.
    :param max_energy: The maximum value for energy. Used for setting x-axis limits.
    :param main_title: The main title for the plot.
    :param save_as: The filename to save the plot as.
    """
    arrival_time, loading_time, energy = data

    # Define the hour ticks and labels
    hours = list(range(0, 25, 3))  # Hours for ticks, every 3 hours
    hour_labels = [str(hour) for hour in hours]  # Convert hours to string labels

    x_data = [arrival_time, arrival_time, loading_time]
    y_data = [loading_time, energy, energy]
    y_limits = [int(max_loading_time), int(max_energy), int(max_energy)]
    y_labels = ['Charging Time [m]', 'Energy [kWh]', 'Energy [kWh]']
    titles = ['Original Data', 'GMM Simulated Data']
    colors = ['b', 'r']
    descriptions = ['Arrival Time vs Charging Time', 'Arrival Time vs Energy', 'Charging Time vs Energy']

    fig, axs = plt.subplots(3, 2, sharey='row', figsize=(8, 8))

    for i in range(3):
        for j in range(2):
            axs[i][j].scatter(x_data[i][j], y_data[i][j], s=1, color=colors[j], label=descriptions[i])
            axs[i][j].legend(loc='upper left')
            axs[i][j].set_xlim(0, MAX_CHARGING_TIME)
            axs[i][j].set_ylim(0, y_limits[i])
            axs[i][j].set_xlabel('Arrival Time [h]' if i < 2 else 'Charging Time [h]')
            if j == 0:
                axs[i][j].set_ylabel(y_labels[i])
            if i == 0:
                axs[i][j].set_title(titles[j])
            axs[i][j].set_xticks([hour * 60 for hour in hours])  # Convert hours to minutes for ticks
            axs[i][j].set_xticklabels(hour_labels)

    fig.suptitle(main_title, fontsize=14, y=0.98, fontweight='bold')

    plt.tight_layout()
    path = 'plots'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{save_as}.png')
    plt.show()


def plot_3d_heatmap(data: list, main_title: str, save_as: str) -> None:
    fig = plt.figure(figsize=(14, 12))

    max_arrival_time = max([max(data[0][0]), max(data[0][1])]) + 10
    min_arrival_time = min([min(data[0][0]), min(data[0][1])]) - 10

    max_loading_time = max([max(data[1][0]), max(data[1][1])]) + 10
    min_loading_time = min([min(data[1][0]), min(data[1][1])]) - 10

    max_energy = max([max(data[2][0]), max(data[2][1])]) + 1
    min_energy = max(min([min(data[2][0]), min(data[2][1])]) - 1, 0)

    # hours = list(range(0, 25, 3))  # Hours for ticks, every 3 hours
    # hour_labels = [str(hour) for hour in hours]  # Convert hours to string labels

    # Process each dataset (original and simulated)
    for i, subplot_title in enumerate(['Original Data', 'GMM Simulated Data']):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')

        # Unpack data
        arrival_times = np.array(data[0][i])
        loading_times = np.array(data[1][i])
        energies = np.array(data[2][i])

        # Creating a meshgrid for the surface plot
        X, Y = np.meshgrid(np.linspace(np.min(arrival_times), np.max(arrival_times), 100),
                           np.linspace(np.min(loading_times), np.max(loading_times), 100))

        # Interpolating the Z values
        Z = griddata((arrival_times, loading_times), energies, (X, Y), method='linear')
        # Interpolate Z values on this grid
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # ax.plot_surface(X, Y, Z, color='b', alpha=0.9)
        ax.set_xlabel('Arrival Time [m]')
        ax.set_ylabel('Charging Time [m]')
        ax.set_zlabel('Energy [kWh]')
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlim(min_arrival_time, max_arrival_time)
        ax.set_ylim(min_loading_time, max_loading_time)
        ax.set_zlim(min_energy, max_energy)
        colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        colorbar.set_label('Charging Demand')
    plt.suptitle(main_title, fontsize=21, y=0.85, fontweight='bold')
    # plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(right=0.975)

    path = 'plots'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{save_as}.png')
    plt.show()


def plot_3d_distribution(data: list, max_loading_time: float, max_energy: float, main_title: str, save_as: str) -> None:
    """
    Plot two 3D scatter plots side by side to show relations between arrival time, loading time, and energy for original and GMM simulated data.

    :param data: A list containing lists of arrival times, loading times, and energy amounts.
    :param max_loading_time: The maximum value for loading time.
    :param max_energy: The maximum value for energy.
    :param main_title: The main title for the plot.
    :param save_as: The filename to save the plot as.
    """
    fig = plt.figure(figsize=(14, 6))

    # Determine the limits for the axes
    max_arrival_time = max([max(data[0][0]), max(data[0][1])])

    # Original Data Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data[0][0], data[1][0], data[2][0], color='b', label='Original Data', alpha=0.5, s=5)
    ax1.set_xlabel('Arrival Time [h]')
    ax1.set_ylabel('Charging Time [h]')
    ax1.set_zlabel('Energy [kWh]')
    ax1.set_title(f'Original Data')
    ax1.legend()
    ax1.set_xlim(0, max_arrival_time)
    ax1.set_ylim(0, max_loading_time)
    ax1.set_zlim(0, max_energy)

    # GMM Simulated Data Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(data[0][1], data[1][1], data[2][1], color='r', label='GMM Simulated Data', alpha=0.5, s=5)
    ax2.set_xlabel('Arrival Time [h]')
    ax2.set_ylabel('Charging Time [h]')
    ax2.set_zlabel('Energy [kWh]')
    ax2.set_title(f'GMM Simulated Data')
    ax2.legend()
    ax2.set_xlim(0, max_arrival_time)
    ax2.set_ylim(0, max_loading_time)
    ax2.set_zlim(0, max_energy)

    plt.suptitle(main_title, fontsize=14, y=0.98, fontweight='bold')

    plt.tight_layout()
    path = 'plots'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{save_as}.png')
    plt.show()


def main():
    n_samples = 10000
    data_set = 'work'

    gmm_file = f'gmm_{data_set}_weekday_70'
    main_title = 'Work - Weekday'
    save_as = 'work_weekday'

    df = get_work_data()

    real_samples = create_real_samples(df, 100000)
    indices = np.random.randint(0, len(real_samples), n_samples)

    real_samples = real_samples[indices]

    real_samples = filter_samples(real_samples)
    real_samples = remove_outliers(real_samples)

    real_arrival_time = real_samples[:, 0].tolist()
    real_loading_time = real_samples[:, 1].tolist()
    real_energy = real_samples[:, 2].tolist()

    generated_arrival_time, generated_loading_time, generated_energy = generate_samples(gmm_file, n_samples)

    max_loading_time = max(np.max(real_loading_time), np.max(generated_loading_time))
    max_energy = max(np.max(real_energy), np.max(generated_energy))

    data = [[real_arrival_time, generated_arrival_time],
            [real_loading_time, generated_loading_time],
            [real_energy, generated_energy]]

    # n_real_data = len(real_samples)
    # X = np.array([generated_arrival_time[:n_real_data], generated_loading_time[:n_real_data], generated_energy[:n_real_data]])
    # Y = np.array([real_arrival_time, real_loading_time, real_energy])
    # result = mkstest(X, Y, alpha=0.05, verbose=True)
    # if result:
    #     print('Samples are NOT drawn from the same distribution')
    # else:
    #     print('Samples are drawn from the same distribution')

    plot_samples(data, max_loading_time, max_energy, f'Histogram: {main_title}', save_as)

    # plot_relations(data, max_loading_time, max_energy, main_title, save_as + '_relation')

    # plot_3d_distribution(data, max_loading_time, max_energy, main_title, save_as + '_3d')
    # plot_3d_heatmap(data, f'Heatmap: {main_title}', save_as + '_heatmap_3d')

    print('Done')


if __name__ == '__main__':
    main()
