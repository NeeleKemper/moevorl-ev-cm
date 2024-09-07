from copy import copy

import numpy as np
import pandas as pd
import gymnasium as gym

from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.spaces import Box

from simulation.ScenarioLoader import ScenarioLoader
from simulation.constant import MAX_PV_POWER, MIN_BATTERY_CAP, MAX_BATTERY_CAP, MIN_CHARGING_POWER, MAX_CHARGING_POWER, \
    MAX_CHARGING_TIME

REWARD_SCALING = 100
PENALTY_SCALING = 1
BASE_PENALTY = -1


class EVChargingTest(gym.Env):
    def __init__(self, scenario_loader: ScenarioLoader, env_id=-1):
        # Initialize variables
        self.scenario_loader = scenario_loader
        self.env_id = env_id
        self.step_counter = -float('inf')

        # Generate scenario and unpack variables
        self.supply, self.event_generator, self.charging_park, self.time_steps = \
            self.scenario_loader.generate_scenario(env_id)

        # Retrieve necessary values from scenario
        scenario = self.scenario_loader.scenario

        self.n_charging_points = scenario['n_charging_points']
        self.action_scaler = scenario['maximum_charging_point_current']
        self.max_phase_difference = scenario['maximum_phase_difference']
        self.min_required_soc = scenario['minimum_required_soc']

        # Calculate power values
        self.max_grid_node_power = scenario['maximum_grid_node_power'] * 1e3
        self.max_pv_power = np.ceil(MAX_PV_POWER * scenario['pv_scaling'])
        self.max_power = self.max_pv_power + self.max_grid_node_power
        self.max_current = self.max_power / ((3 ** 0.5) * 400)

        self.max_phase_difference_violations = 24
        self.max_grid_power_violations = 1000 * self.n_charging_points

        self.pv_power = -1

        # Set state dimensions
        self.n_local_states = 6
        self.n_global_states = 10
        self.n_cp_state = self.n_charging_points * self.n_local_states
        self.n_states = self.n_global_states + self.n_cp_state
        self.observation_space = Box(low=0, high=1, shape=(self.n_states,), dtype=np.float64)

        # Set continues action space
        self.action_space = Box(low=0, high=1.0, shape=(self.n_charging_points,), dtype=np.float64)

        self.reward_dim = 3
        self.reward_space = Box(
            low=np.array(
                [BASE_PENALTY - PENALTY_SCALING, BASE_PENALTY - PENALTY_SCALING, BASE_PENALTY - PENALTY_SCALING]),
            high=np.array([REWARD_SCALING, REWARD_SCALING, REWARD_SCALING]),
            shape=(self.reward_dim,),
            dtype=np.float64
        )

        # soc reward
        self.soc_list = []
        self.ev_counter, self.cumulative_soc_reward = 0, 0
        # load and pv reward
        self.load_profile, self.pv_profile = [], []
        self.soc_profile = None
        # constraints
        self.l1, self.l2, self.l3 = 0, 0, 0
        self.l1_l2, self.l1_l3, self.l2_l3 = 0, 0, 0
        self.ext_grid = 0

        columns_global_states = ['pv_power', 'mod', 'l1', 'l2', 'l3', 'l1_l2', 'l1_l3', 'l2_l3', 'cp_power', 'ext_grid']
        columns_local_states = np.array(
            [[f'state_{cp + 1}', f'soc_{cp + 1}', f'battery_capacity_{cp + 1}', f'charging_power_{cp + 1}',
              f'charging_time_{cp + 1}', f'connected_phases_{cp + 1}'] for cp in
             range(self.n_charging_points)]).flatten()
        columns = columns_global_states + columns_local_states.tolist()
        self.df_obs = pd.DataFrame(columns=columns)

    @staticmethod
    def __clip(value, min_value, max_value):
        return min(max(value, min_value), max_value)

    @staticmethod
    def __decode_charging_phases(connected_phases: np.ndarray) -> int:
        return connected_phases[0] * 4 + connected_phases[1] * 2 + connected_phases[2] * 1

    @staticmethod
    def __normalize(value, value_min, value_max):
        return (value - value_min) / (value_max - value_min)

    def __reset_state(self):
        self.pv_power = -1
        self.step_counter = -float('inf')
        # soc reward
        self.soc_list = []
        self.ev_counter, self.cumulative_soc_reward = 0, 0
        # load and pv reward
        self.load_profile, self.pv_profile = [], []
        self.soc_profile = None
        # constraints
        self.l1, self.l2, self.l3 = 0, 0, 0
        self.l1_l2, self.l1_l3, self.l2_l3 = 0, 0, 0
        self.ext_grid = 0
        self.df_obs = pd.DataFrame(columns=self.df_obs.columns)

    def __get_obs(self, pv_power: float, mod: int) -> np.ndarray:
        self.soc_list = []
        self.pv_profile.append(pv_power)
        if self.pv_power == -1:
            self.pv_power = pv_power

        self.l1, self.l2, self.l3 = self.charging_park.get_i()
        norm_values = [
            self.__normalize(pv_power, 0, self.max_pv_power),
            self.__normalize(mod, 0, (24 * 60) - 1),
            self.__normalize(self.l1, 0, self.max_current),
            self.__normalize(self.l2, 0, self.max_current),
            self.__normalize(self.l3, 0, self.max_current)
        ]

        self.l1_l2, self.l1_l3, self.l2_l3 = map(lambda x: round(abs(x[0] - x[1]), 6),
                                                 [(self.l1, self.l2), (self.l1, self.l3), (self.l2, self.l3)])
        phase_diffs = [self.l1_l2, self.l1_l3, self.l2_l3]
        clipped_diffs = [float(self.__clip(diff, 0, self.max_phase_difference + self.max_phase_difference_violations))
                         for diff in phase_diffs]
        norm_diffs = [self.__normalize(diff, 0, self.max_phase_difference + self.max_phase_difference_violations) for
                      diff in clipped_diffs]

        cp_power = self.charging_park.get_power()
        self.load_profile.append(cp_power)
        norm_load_power = self.__normalize(cp_power, 0, self.max_power)
        self.ext_grid = max(cp_power - self.pv_power, 0)
        clipped_ext_grid = self.__clip(self.ext_grid, 0, self.max_grid_node_power + self.max_grid_power_violations)
        norm_ext_grid = self.__normalize(clipped_ext_grid, 0, self.max_grid_node_power + self.max_grid_power_violations)

        norm_values.extend(norm_diffs)
        norm_values.extend([norm_load_power, norm_ext_grid])

        cps_state = []
        for cp in self.charging_park.get_charging_points():
            if self.charging_park.is_cp_active(cp):
                ev = self.charging_park.get_ev_properties_of_cp(cp)
                self.soc_list.append(ev.soc)
                cp_state = [
                    1,
                    self.__normalize(ev.soc, 0, 1),
                    self.__normalize(ev.battery_capacity, MIN_BATTERY_CAP - 100, MAX_BATTERY_CAP),
                    self.__normalize(ev.max_charging_power, MIN_CHARGING_POWER - 100, MAX_CHARGING_POWER),
                    self.__normalize(ev.charging_time, 0, MAX_CHARGING_TIME),
                    self.__normalize(self.__decode_charging_phases(self.charging_park.get_cp_connected_phases(cp)), 1,
                                     7)
                ]
            else:
                self.soc_list.append(0)
                cp_state = [0, 0, 0, 0, 0, 0]
            cps_state.extend(cp_state)

        if self.soc_profile is None:
            self.soc_profile = self.soc_list
        else:
            self.soc_profile = np.vstack((self.soc_profile, self.soc_list))

        norm_values.extend(cps_state)
        self.pv_power = pv_power
        obs = np.clip(norm_values, 0, 1)
        self.df_obs.loc[len(self.df_obs)] = obs
        return obs

    def __track_soc_rewards(self, departures: dict):
        n = departures['length']
        self.ev_counter += n
        socs, cps = [], []
        for i in range(n):
            soc = self.soc_list[departures['station'][i]]
            socs.append(soc)
            cps.append(departures['station'][i])
            if soc >= self.min_required_soc:
                self.cumulative_soc_reward += 1
            else:
                self.cumulative_soc_reward += (soc / self.min_required_soc) ** 2

    def __soc_reward(self) -> float:
        """
        Compute the SoC reward based on each ev's state of charge.
        :return: The average reward across all evs.
        """
        reward = self.cumulative_soc_reward / self.ev_counter
        return float(self.__clip(reward, 0, 1))

    def __load_reward(self) -> float:
        """
        Compute the load reward based on avoiding peak loads using past loads.
        """

        # Compute the standard deviation reward
        # By using both metrics in this way, the agent gets encourage to keep power consumption smooth (via std_dev)
        # while also avoiding excessive peaks relative to the average (via papr).
        min_load = np.min(self.load_profile)
        max_load = np.max(self.load_profile)

        if max_load - min_load == 0:
            norm_std_dev = 1
        else:
            normalized_load_values = (self.load_profile - min_load) / (max_load - min_load)
            std_dev = np.std(normalized_load_values)
            norm_std_dev = 1 - 2 * std_dev

        # Compute PAPR
        peak_power = max(self.load_profile)
        average_power = sum(self.load_profile) / len(self.load_profile)
        if average_power == 0:
            norm_papr = 0  # Penalize the scenario where no evs are being charged
        else:
            papr = peak_power / average_power
            # Normalize
            norm_papr = 1 / papr

        # Weights for each metric
        weight_std = 0.5
        weight_papr = 0.5
        reward = weight_std * norm_std_dev + weight_papr * norm_papr
        return float(self.__clip(reward, 0, 1))

    def __pv_reward(self) -> float:
        """
        Compute the PV self-consumption reward.
        This function provides a reward between 0 and 1.
        """
        # maximizing PV power consumption and minimizing deviation
        total_pv_power = np.sum(self.pv_profile)
        if total_pv_power == 0:
            return 1

        deviation_penalty = 0
        max_possible_deviation = 0
        for load, pv in zip(self.load_profile, self.pv_profile):
            deviation_penalty += abs(load - pv)
            max_possible_deviation += max(load, pv)

        # Apply penalty for deviation from the PV power
        if max_possible_deviation == 0:
            normalized_penalty = 0
        else:
            normalized_penalty = deviation_penalty / max_possible_deviation
        reward = 1 - normalized_penalty

        return float(self.__clip(reward, 0, 1))

    def __calculate_penalty(self, phase_difference_violations: float, grid_power_violations: float) -> float:
        phase_difference_violations_clipped = float(
            self.__clip(phase_difference_violations, 0, self.max_phase_difference_violations))
        phase_difference_violations_norm = self.__normalize(phase_difference_violations_clipped, 0,
                                                            self.max_phase_difference_violations)

        grid_power_violations_clipped = float(self.__clip(grid_power_violations, 0, self.max_grid_power_violations))
        grid_power_violations_norm = self.__normalize(grid_power_violations_clipped, 0, self.max_grid_power_violations)

        violation_penalty = 0.5 * phase_difference_violations_norm + 0.5 * grid_power_violations_norm
        penalty_reward = BASE_PENALTY - PENALTY_SCALING * violation_penalty
        return penalty_reward

    def __simulation_step(self, step_counter: int):
        # Extract environment values
        pv_power, mod = self.supply.get_supply(step_counter, self.env_id)

        # Determine current time from time_steps
        time = self.time_steps[int(self.step_counter)]

        departures = self.event_generator.get_departures(time)
        arrivals = self.event_generator.get_arrivals(time)

        if departures['length'] > 0:
            self.charging_park.remove_departures(departures)
            self.__track_soc_rewards(departures)

        if arrivals['length'] > 0:
            self.charging_park.assign_arrivals(arrivals)

        return pv_power, mod

    def __check_constraints(self):
        phase_difference = 0
        grid_power = 0
        # Phase difference violations
        if self.l1_l2 > self.max_phase_difference:
            phase_difference += self.l1_l2 - self.max_phase_difference
        if self.l1_l3 > self.max_phase_difference:
            phase_difference += self.l1_l3 - self.max_phase_difference
        if self.l2_l3 > self.max_phase_difference:
            phase_difference += self.l2_l3 - self.max_phase_difference

        # External grid power violation
        if self.ext_grid > self.max_grid_node_power:
            grid_power += self.ext_grid - self.max_grid_node_power

        # Check if any constraint was violated
        constraint_violated = False
        if phase_difference > 0 or grid_power > 0:
            constraint_violated = True
        return constraint_violated, phase_difference, grid_power

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        """Resets the environment to its initial state."""
        self.charging_park.reset()
        self.__reset_state()
        self.step_counter = 0

        pv_power, mod = self.__simulation_step(self.step_counter)
        obs = self.__get_obs(pv_power, mod)
        return obs, {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Takes an action in the environment and returns the next state and associated reward."""

        # continues action
        scaled_action = [round(a * self.action_scaler, 4) for a in action]

        self.charging_park.charging_cycle(actions=scaled_action)

        # Advance to the next state
        self.step_counter += 1
        pv_power, mod= self.__simulation_step(int(self.step_counter))

        next_obs = self.__get_obs(pv_power, mod)

        terminal = False
        # Check terminal state
        if self.step_counter < len(self.time_steps) - 1:
            constraint_violated, phase_difference_violations, grid_power_violations = self.__check_constraints()
            if not constraint_violated:
                rewards = [0, 0, 0]
            else:
                terminal = True
                # Shaped Penalties:
                penalty_reward = self.__calculate_penalty(phase_difference_violations=phase_difference_violations,
                                                          grid_power_violations=grid_power_violations)

                rewards = [penalty_reward, penalty_reward, penalty_reward]


        else:
            # terminal
            terminal = True
            rewards = [self.__soc_reward() * REWARD_SCALING,
                       self.__load_reward() * REWARD_SCALING,
                       self.__pv_reward() * REWARD_SCALING]

        vec_reward = np.array(rewards, dtype=np.float32)

        return next_obs, vec_reward, terminal, False, {}

    def calculate_constraint_metrics(self):
        failure_type = 1
        phase_difference_l1_l2, phase_difference_l1_l3, phase_difference_l2_l3, grid_diff = 0, 0, 0, 0
        if self.l1_l2 > self.max_phase_difference:
            phase_difference_l1_l2 = np.round(self.l1_l2 - self.max_phase_difference, 4)
        if self.l1_l3 > self.max_phase_difference:
            phase_difference_l1_l3 = np.round(self.l1_l3 - self.max_phase_difference, 4)
        if self.l2_l3 > self.max_phase_difference:
            phase_difference_l2_l3 = np.round(self.l2_l3 - self.max_phase_difference, 4)

        # External grid power violation
        if self.ext_grid > self.max_grid_node_power:
            grid_diff = np.round(self.ext_grid - self.max_grid_node_power, 4)
            failure_type = 2

        results = {
            'time_step': int(self.time_steps[int(self.step_counter)]),
            'l1': np.round(self.l1, 4),
            'l2': np.round(self.l2, 4),
            'l3': np.round(self.l3, 4),
            'ext_grid': np.round(self.ext_grid, 4),
            'phase_diff_l1_l2': phase_difference_l1_l2,
            'phase_diff_l1_l3': phase_difference_l1_l3,
            'phase_diff_l2_l3': phase_difference_l2_l3,
            'grid_diff': grid_diff
        }
        df = copy(self.df_obs)
        df['failure'] = 0
        df.at[len(df) - 1, 'failure'] = failure_type
        return results, df

    def calculate_metrics(self):
        soc_mean, soc_std, soc_cov, soc_min, soc_max, charged, charged_amount_mean, target_reached, target_diff_mean \
            = self.__calculate_soc_metrics()
        self_consumed_mean, ext_mean, feed_mean, self_consumption_rate, feed_back_rate, grid_dependency = self.__calculate_pv_metrics()
        papr, load_std, load_mean, load_sum, max_load, load_factor, load_variability_index = self.__calculate_load_metrics()
        results = {
            'soc_mean': soc_mean,  # soc
            'soc_std': soc_std,
            'soc_cov': soc_cov,  # soc
            'soc_min': soc_min,
            'soc_max': soc_max,
            'charged': charged,
            'charged_amount_mean': charged_amount_mean,
            'target_reached': target_reached,  # soc
            'target_diff_mean': target_diff_mean,

            'self_consumed_mean': self_consumed_mean,  # pv
            'ext_mean': ext_mean,
            'feed_mean': feed_mean,
            'self_consumption_rate': self_consumption_rate,
            'feed_back_rate': feed_back_rate,  # pv
            'grid_dependency': grid_dependency,  # pv

            'papr': papr,  # load
            'load_std': load_std,
            'load_mean': load_mean,
            'load_sum': load_sum,
            'max_load': max_load,
            'load_factor': load_factor,  # load
            'lvi': load_variability_index  # load
        }
        return results

    def __calculate_soc_metrics(self):
        columns_as_lists = [self.soc_profile[:, i].tolist() for i in range(self.soc_profile.shape[1])]
        soc_diffs, soc_final, target_diff = [], [], []
        for col in columns_as_lists:
            for soc_tuple in self.__find_non_zero_sequences(col):
                soc_diffs.append(np.round(soc_tuple[1] - soc_tuple[0], 4))
                soc_final.append(np.round(soc_tuple[1], 4))
                target_diff.append(max(np.round(self.min_required_soc - soc_tuple[1], 4), 0))

        charged = np.round(1 - soc_diffs.count(0) / len(soc_diffs), 4)
        charged_amount_mean = np.round(np.mean(soc_diffs), 4)
        target_reached = np.round(sum(1 for soc in soc_final if soc >= self.min_required_soc) / len(soc_final), 4)
        target_diff_mean = np.round(np.mean(target_diff), 4)

        soc_mean = np.round(np.mean(soc_final), 4)
        soc_std = np.round(np.std(soc_final), 4)
        soc_cov = np.round(soc_std / soc_mean, 4)
        soc_min = np.round(np.min(soc_final), 4)
        soc_max = np.round(np.max(soc_final), 4)
        return soc_mean, soc_std, soc_cov, soc_min, soc_max, charged, charged_amount_mean, target_reached, target_diff_mean

    def __calculate_pv_metrics(self):
        pv_profile = [pv / 1000 for pv in self.pv_profile]
        load_profile = [load / 1000 for load in self.load_profile]

        self_consumed_profile = [min(pv, load) for pv, load in zip(pv_profile, load_profile)]
        feed_profile = [max(0, pv - load) for pv, load in zip(pv_profile, load_profile)]
        ext_profile = [max(0, load - pv) for pv, load in zip(pv_profile, load_profile)]

        total_pv_power = sum(pv_profile)
        total_load = sum(load_profile)

        if total_pv_power == 0:
            self_consumption_rate = 1
            feed_back_rate = 0
        else:
            self_consumption_rate = np.round(sum(self_consumed_profile) / total_pv_power, 4)
            feed_back_rate = np.round(sum(feed_profile) / total_pv_power, 4)
        if total_load == 0:
            grid_dependency = 0
        else:
            grid_dependency = np.round(sum(ext_profile) / total_load, 4)

        self_consumed_mean = np.round(np.mean(self_consumed_profile), 4)
        ext_mean = np.round(np.mean(ext_profile), 4)
        feed_mean = np.round(np.mean(feed_profile), 4)
        return self_consumed_mean, ext_mean, feed_mean, self_consumption_rate, feed_back_rate, grid_dependency

    def __calculate_load_metrics(self):
        load_profile = [load / 1000 for load in self.load_profile]

        min_load = np.round(np.min(load_profile), 4)
        max_load = np.round(np.max(load_profile), 4)
        if max_load - min_load == 0:
            papr, load_std, load_mean, load_sum, load_factor, load_variability_index = 0, 0, 0, 0, 0, 0
        else:
            load_mean = np.round(np.mean(load_profile), 4)
            papr = np.round(max_load / load_mean, 4)
            load_std = np.round(np.std(load_profile), 4)
            load_sum = np.round(np.sum(load_profile), 4)
            load_factor = np.round(load_mean / max_load, 4)
            load_variability_index = np.round(load_std / load_mean, 4)
        return papr, load_std, load_mean, load_sum, max_load, load_factor, load_variability_index

    @staticmethod
    def __find_non_zero_sequences(lst):
        sequences = []
        start = None

        for i in range(len(lst)):
            if lst[i] != 0 and start is None:
                start = lst[i]  # Start of a new sequence
            elif lst[i] == 0 and start is not None:
                sequences.append((start, lst[i - 1]))
                # End of the current sequence
                start = None  # Reset for the next sequence

        # Check if the last sequence goes till the end of the list
        if start is not None:
            sequences.append((start, lst[-1]))
        return sequences


def render(self) -> RenderFrame | list[RenderFrame] | None:
    pass


def pareto_front(self, gamma: float):
    pass
