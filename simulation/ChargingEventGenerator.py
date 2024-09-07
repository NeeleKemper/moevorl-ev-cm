import os
import pickle
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choice
from typing import Tuple
from datetime import datetime, timedelta
from data.charging_event.charging_events_utils import filter_samples
from simulation.constant import CHARGING_INTERVAL, START_DATE

sns.set()


def modify_phase_strings(car_names):
    return car_names.replace(' 1-phase', ' (1-phase)').replace(' 2-phase', ' (2-phase)').replace(' 3-phase',
                                                                                                 ' (3-phase)')


def get_plot_title(scenario, charging_density, seed):
    return f'{scenario} (Density={charging_density}, EnvSeed={seed})'


def plot_car_counts(df: pd.DataFrame, scenario: str, path: str, charging_density: str, seed: int) -> None:
    car_names = df.index.tolist()
    modified_car_names = [modify_phase_strings(name) for name in car_names]

    car_counts = df.values.tolist()  # Assuming 'Count' is the column name

    count_1_phase, count_2_phase, count_3_phase = 0, 0, 0
    for name, count in zip(car_names, car_counts):
        if '1-phase' in name:
            count_1_phase += count
        elif '2-phase' in name:
            count_2_phase += count
        else:
            count_3_phase += count

    print(f'1-Phase: {count_1_phase}, 2-Phase: {count_2_phase}, 3-Phase: {count_3_phase}')

    plt.figure(figsize=(12, 6))
    plt.barh(modified_car_names, car_counts, height=0.75)
    plt.xlabel('Count')
    plt.title(f'{get_plot_title(scenario, charging_density, seed)}: Car Count by Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # To display the highest count at the top
    plt.tight_layout()
    # title = title.replace(' ', '_')
    # plt.savefig(f'{path}/{scenario}_car_count_{charging_density}_{seed}.png')
    plt.show()

def plot_distributions(df: pd.DataFrame, scenario: str, path: str, charging_density: str, seed: int) -> None:
    """
    Plots the distribution of arrival times, charging time, and SOC, separated by weekday and weekend.
    """
    # Calculate additional columns
    df['hour_of_day'] = (df['arrival_time'] % 1440) / 60
    df['charging_time_hours'] = df['charging_time'] / 60
    df['weekday'] = df['weekday'].astype(bool)

    # Filter data for weekdays and weekends
    weekdays = df[df['weekday']]
    weekends = df[~df['weekday']]

    # Decide on the subplot layout based on available data
    num_columns = 2 if len(weekdays) > 0 and len(weekends) > 0 else 1

    fig, axes = plt.subplots(nrows=3, ncols=num_columns, figsize=(8, 8), sharey='row')

    # If the subplot has only one column, axes will be 1D, so we convert it to a 2D array with one column
    if num_columns == 1:
        axes = np.expand_dims(axes, axis=1)

    titles = ['Weekday'] if len(weekends) == 0 else ['Weekend'] if len(weekdays) == 0 else ['Weekday', 'Weekend']
    data_sets = [weekdays] if len(weekends) == 0 else [weekends] if len(weekdays) == 0 else [weekdays, weekends]

    # Plot each of the distributions
    for i, data_set in enumerate(data_sets):
        # Arrival Time
        ax = axes[0][i]  # Access with single index since axes is now 2D
        ax.hist(data_set['hour_of_day'], bins=48, edgecolor='k', alpha=0.7, label='Arrival Time')
        # ax.set_title(f'{titles[i]}')
        ax.set_xlabel('Time [h]')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))  # This will set the ticks at every 2 hours
        ax.set_xticklabels([f'{hour}' for hour in range(0, 25, 2)])
        ax.legend()
        if i == 0:
            ax.set_ylabel('Count')

        # Charging Time
        ax = axes[1][i]
        ax.hist(data_set['charging_time_hours'], bins=24, edgecolor='k', alpha=0.7, label='Charging Time')
        ax.set_xlabel('Time [h]')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))  # This will set the ticks at every 2 hours
        ax.set_xticklabels([f'{hour}' for hour in range(0, 25, 2)])
        ax.legend()
        if i == 0:
            ax.set_ylabel('Count')

        # State of Charge (SoC)
        ax = axes[2][i]
        ax.hist(data_set['soc'] * 100, bins=25, edgecolor='k', alpha=0.7, label='SoC')
        print(f'SoC mean (std): {round(data_set["soc"].mean(),2)}, ({round(data_set["soc"].std(),2)})')
        print(f'SoC min/max: {round(data_set["soc"].min(),2)} / {round(data_set["soc"].max(),2)}')

        ax.set_xlabel('State of Charge [%]')
        ax.set_xlim(0, 100)
        ax.legend()
        if i == 0:
            ax.set_ylabel('Count')

    # Set x-ticks for the bottom row of plots
    for ax in axes[2]:  # No need to check for the number of columns here
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticklabels([f'{i}' for i in range(0, 101, 10)])

    # Tight layout to ensure the subplots fit into the figure area nicely
    plt.tight_layout(pad=3.0)

    # Adjust the overall title and display the plot
    plt.suptitle(f'{get_plot_title(scenario, charging_density, seed)}: Distributions', fontsize=14, y=0.98,
                 fontweight='bold')
    # title = title.replace(' ', '_')
    # plt.savefig(f'{path}/{scenario}_distribution_{charging_density}_{seed}.png')
    plt.show()


class ChargingEventGenerator(object):
    def __init__(self, n_charging_points: int,  max_charging_point_current: int,
                 density: float, density_note: str, minimum_required_soc: float,  seed: int,
                 scenario: str) -> None:

        # Set seed for random number generation for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        self.random_seed = seed

        # Define the number of charging points and their density
        self.n_charging_points = n_charging_points
        self.density = density
        self.density_note = density_note
        self.minimum_required_soc = minimum_required_soc
        self.max_charging_point_current = max_charging_point_current
        self.n_days = 365
        # Dummy start date (start weekday is monday)
        self.start_date = START_DATE

        # Load the list of electric vehicles
        self.cars = json.load(open('electric_vehicles.json'))
        self.car_selection_counts = {car: 0 for car in self.cars.keys()}  # Initialize counts for each car

        gmm_file_weekday = f'gmm_work_weekday_70'

        # Generate samples using GMM models
        self.samples_weekday = self.__generate_samples(gmm_file_weekday)

        self.max_energy = np.max(self.samples_weekday[:, 2])
        self.min_energy = np.min(self.samples_weekday[:, 2])

        # Draft a timetable for charging events
        self.schedule = self.__draft_timetable()
        self.schedule['arrival_time_date'] = pd.to_datetime(self.schedule['arrival_time_date'])
        print('Average number of events per day: ',
              round(self.schedule.groupby(self.schedule['arrival_time_date'].dt.date).count().mean()['car'], 2))
        self.arrivals, self.departures = self.dataframe_to_dict(self.schedule)
        car_counts = self.schedule['car'].value_counts()
        print(car_counts)

        path = 'simulation/plots'
        if not os.path.isdir(path):
            os.makedirs(path)
        plot_distributions(self.schedule, scenario, path, self.density_note, seed)
        plot_car_counts(car_counts, scenario, path, self.density_note, seed)

    @staticmethod
    def dataframe_to_dict(df):
        arrival_dict = defaultdict(list)
        departure_dict = defaultdict(list)

        for _, row in df.iterrows():
            arrival_info = {
                'station': row['station'],
                'charging_time': row['charging_time'],
                'soc': row['soc'],
                'car': row['car']
            }
            departure_info = {'station': row['station']}

            arrival_dict[row['arrival_time']].append(arrival_info)
            departure_dict[row['departure_time']].append(departure_info)

        return arrival_dict, departure_dict

    @staticmethod
    def __generate_samples(gmm_file: str) -> np.ndarray:
        """
        Generate samples using Gaussian Mixture Models (GMM)
        :param gmm_file: File name of the saved GMM model
        :return: An array of generated samples and the minimum and maximum loading time of the samples
        """
        with open(f'data/charging_event/models/{gmm_file}.pkl', 'rb') as f:
            gmm_model_weekday = pickle.load(f)

            # Generate 1e7 samples using the GMM model
            generated_samples, _ = gmm_model_weekday.sample(1e7)

            # Filter samples to keep only those meeting specific conditions
            generated_samples = filter_samples(generated_samples)

        return generated_samples

    def __draft_timetable(self) -> pd.DataFrame:
        """
        Draft a timetable for charging events.
        :return: A DataFrame with the timetable.
        """
        # Initialize a DataFrame to store the timetable
        timetable = pd.DataFrame(columns=['station', 'arrival_time', 'departure_time', 'charging_time', 'soc', 'car'])

        # Initialize queue for stations
        stations = [(self.start_date, i) for i in range(0, self.n_charging_points)]

        for day in range(self.n_days):
            # Calculate the number of samples needed for a day
            if self.density_note == 'low':
                n_samples = int(self.density / 2 * self.n_charging_points)
            elif self.density_note == 'high':
                n_samples = int(self.density * 15 * self.n_charging_points)
            else:
                n_samples = int(self.density * self.n_charging_points)

            # Calculate the current date
            current_date = self.start_date + timedelta(days=day)
            # Check if any station is still busy from the previous day
            stations = [[max(available_time, current_date), station] for available_time, station in stations]
            # Set seed for the current day
            np.random.seed(self.random_seed + day)
            random.seed(self.random_seed + day)

            # Draw samples depending day type or on whether the current day is a weekday or weekend
            samples = self.__draw_samples(self.samples_weekday, n_samples)

            if samples is not None:
                #  Sort the charging sessions by arrival time
                if self.density_note == 'high':
                    if random.uniform(0, 1) > 0.33:
                        sorted_samples = samples  # [samples[:, 0].argsort()]
                    else:
                        sorted_samples = samples[samples[:, 0].argsort()]
                else:
                    sorted_samples = samples[samples[:, 0].argsort()]

                # Schedule charging events
                for arrival_time, charging_time, energy in sorted_samples:
                    if self.density_note == 'high':
                        arrival_time += np.random.uniform(0, 30)

                    # Compute the datetime of arrival and convert to string in 'YYYY-MM-DD HH:MM' format
                    arrival_time = self.__compute_arrival_time(current_date, arrival_time)
                    departure_time = self.__compute_departure_time(arrival_time, charging_time)

                    # Find the stations that are available at the arrival_time
                    available_stations = [
                        idx for idx, station in enumerate(stations)
                        if station[0] <= datetime.strptime(arrival_time, '%Y-%m-%d %H:%M')
                    ]

                    # If any stations are available, select one at random
                    if available_stations:
                        chosen_idx = choice(available_stations)
                        available_station = stations[chosen_idx][1]
                        # Update the chosen station's available time
                        stations[chosen_idx][0] = datetime.strptime(departure_time, '%Y-%m-%d %H:%M') + timedelta(
                            minutes=1)
                    else:
                        available_station = None

                    if available_station is not None:
                        # Draw a state of charge (SOC) value and select a car
                        soc, car = self.__draw_soc_ev(energy, charging_time)

                        # Add the new charging event to the timetable
                        timetable = pd.concat([timetable, pd.DataFrame([{
                            'station': available_station,
                            'arrival_time': arrival_time,
                            'departure_time': departure_time,
                            'charging_time': int(charging_time),
                            'soc': soc,
                            'car': car,
                            'weekday': True
                        }])], ignore_index=True)

        timetable = self.__timestamps_to_time_steps(timetable)
        print(f'Number of scheduled charging events: {len(timetable)}')
        return timetable

    def __timestamps_to_time_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamped data to time steps using a datetime index.

        :param df: Input DataFrame with 'arrival_time' and 'departure_time' columns.
        :return: Modified DataFrame where 'arrival_time' and 'departure_time' are replaced with their corresponding time steps.
        """

        end_date = self.start_date + pd.Timedelta(self.n_days, 'D') - pd.Timedelta(1, 'S')
        dt_index = pd.date_range(start=self.start_date, end=end_date, freq=f'{CHARGING_INTERVAL}min')
        date_list = dt_index.tolist()
        date_to_index = {date: index for index, date in enumerate(date_list)}
        df['arrival_time_date'] = df['arrival_time']
        df['departure_time_date'] = df['departure_time']
        df['arrival_time'] = df['arrival_time'].map(date_to_index)
        df['departure_time'] = df['departure_time'].map(date_to_index)
        return df

    @staticmethod
    def __is_weekday(date: datetime) -> bool:
        """
        Check if a date is a weekday.
        :param date: The date to check.
        :return: True if the date is a weekday, False otherwise.
        """
        # The weekday method returns the day of the week as an integer (Monday=0, Sunday=6)
        if date.weekday() < 5:
            return True
        return False

    @staticmethod
    def __draw_samples(gmm_samples: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Draw samples from an array.
        :param gmm_samples: The array to draw samples from.
        :param n_samples: The number of samples to draw.
        :return: An array of drawn samples.
        """
        # Randomly select indices of the sample_array to draw
        indices = np.random.randint(0, len(gmm_samples), n_samples)
        # Return the drawn samples
        return gmm_samples[indices]

    @staticmethod
    def __compute_arrival_time(current_date: datetime, arrival_time: float) -> str:
        """
        Compute the arrival time as a datetime string.
        :param current_date: The current date.
        :param arrival_time: The arrival time in minutes.
        :return: The arrival time as a datetime string in 'YYYY-MM-DD HH:MM' format.
        """
        # Compute the arrival time as a timedelta
        arrival_delta = timedelta(minutes=int(arrival_time))
        # Add the arrival time to the current date and return as a string
        return (current_date + arrival_delta).strftime('%Y-%m-%d %H:%M')

    @staticmethod
    def __compute_departure_time(arrival_time: str, charging_time: float) -> str:
        """
        Compute the departure time as a datetime string.
        :param arrival_time: The arrival time as a datetime string in 'YYYY-MM-DD HH:MM' format.
        :param charging_time: The loading time in minutes.
        :return: The departure time as a datetime string in 'YYYY-MM-DD HH:MM' format.
        """
        # Convert the arrival time to a datetime
        arrival_time = datetime.strptime(arrival_time, '%Y-%m-%d %H:%M')
        # Compute the departure time as a timedelta
        departure_delta = timedelta(minutes=int(charging_time))
        # Add the departure time to the arrival time and return as a string
        return (arrival_time + departure_delta).strftime('%Y-%m-%d %H:%M')

    def __draw_soc_ev(self, energy: float, charging_time: int) -> \
            Tuple[float, str]:
        """
        Draw a state of charge (SOC) value and select a car based on provided energy and loading time.

        The method also employs a fallback mechanism to select a suitable car if the initial conditions are not met.

        :param energy: The energy required for charging.
        :param charging_time: The duration of the charging session in minutes.
        :return: A tuple containing the drawn SOC value and the selected car.
        """
        cars_info = []

        max_soc = self.minimum_required_soc
        min_soc = 0.1
        offset = 0.1
        for car, details in self.cars.items():
            charging_power = details[f'charging_power_{self.max_charging_point_current}A']
            energy_added = charging_power * (charging_time / 60)

            if energy_added >= details['battery_capacity_kwh']:
                soc = 1  # Battery is fully charged
            else:
                soc = energy_added / details['battery_capacity_kwh']
            soc = 1 - soc
            cars_info.append((soc, car))

        base_soc = min_soc + (
                ((max_soc - offset) - min_soc) * (self.max_energy - energy) / (self.max_energy - self.min_energy))

        # Compute scores for each car. Lower scores are better.
        scores = [abs(car[0] - base_soc) for car in cars_info]

        # Convert scores to probabilities. Incorporate selection count.
        # Penalize cars that have been selected more times
        selection_penalties = [self.car_selection_counts[car[1]] + 1 for car in
                               cars_info]  # +1 to avoid division by zero

        probabilities = [(1 / s) / sp if s != 0 else 1 / sp for s, sp in zip(scores, selection_penalties)]
        total_prob = sum(probabilities)
        normalized_probabilities = [p / total_prob for p in probabilities]

        # Randomly select an index based on the computed probabilities
        index = np.random.choice(len(cars_info), p=normalized_probabilities)

        # Use the selected index to retrieve the corresponding car
        car = cars_info[index]

        _, model = car
        # Increment the selection count for the chosen car
        self.car_selection_counts[model] += 1

        soc = np.clip(base_soc, min_soc + offset, max_soc - offset) + random.gauss(-0.05, 0.025)
        soc = np.clip(soc, min_soc, max_soc - offset)
        return soc, model

    def get_arrivals(self, time_step: int) -> dict:
        """
        Retrieves all arrivals at a specific time step.

        :param time_step: Time step to filter arrivals.
        :return: Dictionary containing arrival information for the specific time step.
        """
        arrivals_at_time = self.arrivals.get(time_step, [])
        return {
            'station': [entry['station'] for entry in arrivals_at_time],
            'charging_time': [entry['charging_time'] for entry in arrivals_at_time],
            'soc': [entry['soc'] for entry in arrivals_at_time],
            'car': [entry['car'] for entry in arrivals_at_time],
            'length': len(arrivals_at_time)
        }

    def get_departures(self, time_step: int) -> dict:
        departures_at_time = self.departures.get(time_step, [])
        return {
            'station': [entry['station'] for entry in departures_at_time],
            'length': len(departures_at_time)
        }
