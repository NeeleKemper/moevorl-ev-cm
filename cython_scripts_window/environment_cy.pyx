# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False,  language_level=3
import math

from libc.math cimport fabs
# noinspection PyUnresolvedReferences
from cython_scripts.charging_park_cy import ChargingPark

cdef class Environment:
    def __init__(self, ChargingPark charging_park, list supply, dict departures, dict arrivals, int env_id,
                 double action_scaler,
                 list time_steps, double min_required_soc, double max_power, double max_pv_power,
                 double max_grid_node_power, double max_current, double max_phase_difference, double max_charging_time,
                 double min_battery_cap, double max_battery_cap, double min_charging_power, double max_charging_power,
                 int n_charging_points):
        self.charging_park = charging_park

        self.supply = supply
        self.departures = departures
        self.arrivals = arrivals

        self.env_id = env_id
        self.action_scaler = action_scaler
        self.time_steps = time_steps
        self.max_steps = len(time_steps)
        self.min_required_soc = min_required_soc
        self.max_power = max_power
        self.max_pv_power = max_pv_power
        self.max_grid_node_power = max_grid_node_power
        self.max_current = max_current
        self.max_phase_difference = max_phase_difference
        self.max_charging_time = max_charging_time
        self.min_battery_cap = min_battery_cap
        self.max_battery_cap = max_battery_cap
        self.min_charging_power = min_charging_power
        self.max_charging_power = max_charging_power
        self.n_charging_points = n_charging_points

        self.step_counter = 0
        self.pv_power = -1
        self.soc_list = [0.0] * self.n_charging_points
        self.pv_profile = [0.0] * self.max_steps
        self.load_profile = [0.0] * self.max_steps
        self.car_counter, self.cumulative_soc_reward = 0, 0
        self.l1_l2, self.l1_l3, self.l2_l3 = 0, 0, 0
        self.ext_grid = 0

        self.max_phase_difference_violations = 24
        self.max_grid_power_violations = 1000 * self.n_charging_points

        self.REWARD_SCALING = 100.0
        self.PENALTY_SCALING = 1.0
        self.BASE_PENALTY = -1.0

    cdef void __reset_state(self):
        self.step_counter = 0
        self.pv_power = -1
        self.soc_list = [0.0] * self.n_charging_points
        self.pv_profile = [0.0] * self.max_steps
        self.load_profile = [0.0] * self.max_steps
        self.car_counter, self.cumulative_soc_reward = 0, 0
        self.l1_l2, self.l1_l3, self.l2_l3 = 0, 0, 0
        self.ext_grid = 0

    cdef inline double __decode_charging_phases(self, list connected_phases):
        return connected_phases[0] * 4 + connected_phases[1] * 2 + connected_phases[2] * 1

    cdef inline double __normalize(self, double value, double value_min, double value_max) nogil:
        return (value - value_min) / (value_max - value_min)

    cdef inline double __clip(self, double value, double min_value, double max_value) nogil:
        return min(max(value, min_value), max_value)

    cdef list __get_obs(self, double pv_power, int mod, int step):
        self.soc_list = [0.0] * self.n_charging_points
        self.pv_profile[step] = pv_power
        if self.pv_power == -1:
            self.pv_power = pv_power

        cdef double l1, l2, l3
        l1, l2, l3 = self.charging_park.get_i()

        norm_values = [
            self.__normalize(pv_power, 0, self.max_pv_power),
            self.__normalize(mod, 0, (24 * 60) - 1),
            self.__normalize(l1, 0, self.max_current),
            self.__normalize(l2, 0, self.max_current),
            self.__normalize(l3, 0, self.max_current)
        ]

        self.l1_l2 = fabs(l1 - l2)
        self.l1_l3 = fabs(l1 - l3)
        self.l2_l3 = fabs(l2 - l3)

        phase_diffs = [self.l1_l2, self.l1_l3, self.l2_l3]
        # clipped_diffs = [self.__clip(x, 0, self.max_phase_difference + self.action_scaler) for x in phase_diffs]
        phase_violations = [max(0, x - self.max_phase_difference) for x in phase_diffs]
        clipped_phase_violations = [self.__clip(x, 0, self.max_phase_difference_violations) for x in phase_diffs]
        norm_diffs = [self.__normalize(x, 0, self.max_phase_difference_violations) for x in phase_violations]
        norm_values.extend(norm_diffs)

        cdef double cp_power = self.charging_park.get_power()
        self.load_profile[step] = cp_power
        norm_load_power = self.__normalize(cp_power, 0, self.max_power)
        self.ext_grid = max(cp_power - self.pv_power, 0)

        # clipped_ext_grid = self.__clip(self.ext_grid, 0, self.max_grid_node_power + self.ext_grid_offset)
        grid_violation = max(0, self.ext_grid - self.max_grid_node_power)
        clipped_grid_violation = self.__clip(grid_violation, 0, self.max_grid_power_violations)
        norm_ext_grid = self.__normalize(clipped_grid_violation, 0, self.max_grid_power_violations)

        norm_values.extend([norm_load_power, norm_ext_grid])

        cps = self.charging_park.get_charging_points()
        cps_state = []

        for i, cp in enumerate(cps):
            if self.charging_park.is_cp_active(cp):
                ev = self.charging_park.get_ev_properties_of_cp(cp)
                self.soc_list[i] = ev.soc
                cp_state = [
                    1,
                    self.__normalize(ev.soc, 0, 1),
                    self.__normalize(ev.battery_capacity, self.min_battery_cap, self.max_battery_cap),
                    self.__normalize(ev.max_charging_power, self.min_charging_power, self.max_charging_power),
                    self.__normalize(ev.charging_time, 0, self.max_charging_time),
                    self.__normalize(self.__decode_charging_phases(self.charging_park.get_cp_connected_phases(cp)), 1,
                                     7)
                ]
            else:
                self.soc_list[i] = 0
                cp_state = [0, 0, 0, 0, 0, 0]
            cps_state.extend(cp_state)
        self.pv_power = pv_power
        norm_values.extend(cps_state)
        return norm_values

    cdef double __calculate_penalty(self, double phase_difference_violations, double grid_power_violations):
        cdef double phase_difference_violations_clipped = self.__clip(phase_difference_violations, 0,
                                                                      self.max_phase_difference_violations)
        cdef double phase_difference_violations_norm = self.__normalize(phase_difference_violations_clipped, 0,
                                                                        self.max_phase_difference_violations)

        cdef double grid_power_violations_clipped = self.__clip(grid_power_violations, 0,
                                                                self.max_grid_power_violations)
        cdef double grid_power_violations_norm = self.__normalize(grid_power_violations_clipped, 0,
                                                                  self.max_grid_power_violations)

        cdef double violation_penalty = 0.5 * phase_difference_violations_norm + 0.5 * grid_power_violations_norm

        cdef double penalty_reward = self.BASE_PENALTY - self.PENALTY_SCALING * violation_penalty
        return penalty_reward

    cdef double __load_reward(self, list load_profile):
        cdef int n = len(load_profile)
        cdef double min_load = min(load_profile)
        cdef double max_load = max(load_profile)
        cdef double norm_std_dev
        cdef double std_dev
        cdef double peak_power
        cdef double average_power
        cdef double papr
        cdef double norm_papr
        cdef double weight_std = 0.5
        cdef double weight_papr = 0.5
        cdef double reward
        cdef list normalized_load_values = [0.0] * n
        cdef int i
        cdef double sum_load = 0
        cdef double mean

        if max_load - min_load == 0:
            norm_std_dev = 1
        else:
            for i in range(n):
                normalized_load_values[i] = (load_profile[i] - min_load) / (max_load - min_load)
                sum_load += normalized_load_values[i]

            mean = sum_load / n
            std_dev = 0
            for i in range(n):
                std_dev += (normalized_load_values[i] - mean) ** 2
            std_dev = (std_dev / n) ** 0.5
            norm_std_dev = 1 - 2 * std_dev

        peak_power = max(load_profile)
        average_power = sum(load_profile) / n
        if average_power == 0:
            norm_papr = 0
        else:
            papr = peak_power / average_power
            norm_papr = 1 / papr

        reward = weight_std * norm_std_dev + weight_papr * norm_papr
        return self.__clip(reward, 0, 1)

    cdef double __pv_reward(self, list pv_profile, list load_profile):
        cdef double total_pv_power = sum(pv_profile)
        cdef double deviation_penalty = 0
        cdef double max_possible_deviation = 0
        cdef double load
        cdef double pv
        cdef double normalized_penalty
        cdef double reward

        if total_pv_power == 0:
            return 1

        for load, pv in zip(load_profile, pv_profile):
            deviation_penalty += fabs(load - pv)
            max_possible_deviation += max(load, pv)

        if max_possible_deviation == 0:
            normalized_penalty = 0
        else:
            normalized_penalty = deviation_penalty / max_possible_deviation

        reward = 1 - normalized_penalty
        return self.__clip(reward, 0, 1)

    cpdef void __track_soc_rewards(self, dict departures):
        cdef int n = departures['length']
        self.car_counter += n

        cdef int i
        cdef double soc
        for i in range(n):
            soc = self.soc_list[departures['station'][i]]
            if soc >= self.min_required_soc:
                self.cumulative_soc_reward += 1.0
            else:
                self.cumulative_soc_reward += (soc / self.min_required_soc) ** 2

    cpdef double __soc_reward(self):
        cdef double reward = self.cumulative_soc_reward / self.car_counter
        return self.__clip(reward, 0, 1)

    cdef tuple __check_constraints(self):
        cdef double phase_difference = 0
        cdef double grid_power = 0
        cdef bint constraint_violated

        if self.l1_l2 > self.max_phase_difference:
            phase_difference += self.l1_l2 - self.max_phase_difference
        if self.l1_l3 > self.max_phase_difference:
            phase_difference += self.l1_l3 - self.max_phase_difference
        if self.l2_l3 > self.max_phase_difference:
            phase_difference += self.l2_l3 - self.max_phase_difference

        if self.ext_grid > self.max_grid_node_power:
            grid_power += self.ext_grid - self.max_grid_node_power

        constraint_violated = phase_difference > 0 or grid_power > 0

        return constraint_violated, phase_difference, grid_power

    cdef tuple __get_supply(self, int time_step, int env_id):
        if env_id == -1:
            row = self.supply[time_step]
        else:
            env = self.supply[env_id]
            row = env[time_step]

        return row[1], row[2]

    cdef dict __get_arrivals(self, int time_step):
        arrivals_at_time = self.arrivals.get(time_step, [])
        return {
            'station': [entry['station'] for entry in arrivals_at_time],
            'charging_time': [entry['charging_time'] for entry in arrivals_at_time],
            'soc': [entry['soc'] for entry in arrivals_at_time],
            'car': [entry['car'] for entry in arrivals_at_time],
            'length': len(arrivals_at_time)
        }

    cdef dict __get_departures(self, int time_step):
        departures_at_time = self.departures.get(time_step, [])
        return {
            'station': [entry['station'] for entry in departures_at_time],
            'length': len(departures_at_time)
        }

    cdef tuple __simulation_step(self, int step_counter):
        # Extract environment values
        pv_power, mod = self.__get_supply(step_counter, self.env_id)

        # Determine current time from time_steps
        time = self.time_steps[int(self.step_counter)]

        departures = self.__get_departures(time)
        arrivals = self.__get_arrivals(time)

        if departures['length'] > 0:
            self.charging_park.remove_departures(departures)
            self.__track_soc_rewards(departures)

        if arrivals['length'] > 0:
            self.charging_park.assign_arrivals(arrivals)

        return pv_power, mod

    cpdef tuple reset(self):
        self.charging_park.reset()
        self.__reset_state()
        self.step_counter = 0
        pv_power, mod = self.__simulation_step(self.step_counter)
        obs = self.__get_obs(pv_power, mod, self.step_counter)
        return obs, {}

    cpdef tuple step(self, list action):
        # continues action
        cdef list scaled_action = [a * self.action_scaler for a in action]
        self.charging_park.charging_cycle(actions=scaled_action)

        # Advance to the next state
        self.step_counter += 1
        pv_power, mod = self.__simulation_step(int(self.step_counter))

        next_obs = self.__get_obs(pv_power, mod, self.step_counter)

        cdef bint terminal = False
        # Check terminal state
        if self.step_counter < self.max_steps - 1:
            constraint_violated, phase_difference_violations, grid_power_violations = self.__check_constraints()
            if not constraint_violated:
                rewards = [0., 0., 0.]

            else:
                terminal = True
                # Shaped Penalties:
                penalty_reward = self.__calculate_penalty(phase_difference_violations=phase_difference_violations,
                                                          grid_power_violations=grid_power_violations)

                rewards = [penalty_reward, penalty_reward, penalty_reward]


        else:
            # terminal
            terminal = True
            rewards = [
                self.__soc_reward() * self.REWARD_SCALING,
                self.__load_reward(self.load_profile) * self.REWARD_SCALING,
                self.__pv_reward(self.pv_profile, self.load_profile) * self.REWARD_SCALING,
            ]

        return next_obs, rewards, terminal, False, {}
