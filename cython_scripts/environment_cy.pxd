# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False,  language_level=3
cimport numpy as cnp
from .charging_park_cy cimport ChargingPark

cdef class Environment:
    cdef readonly dict departures
    cdef readonly dict arrivals
    cdef readonly int env_id
    cdef readonly double action_scaler
    cdef readonly list time_steps
    cdef readonly int max_steps
    cdef readonly double min_required_soc
    cdef readonly double max_power
    cdef readonly double max_pv_power
    cdef readonly double max_grid_node_power
    cdef readonly double max_current
    cdef readonly double max_phase_difference
    cdef readonly double max_charging_time
    cdef readonly double min_battery_cap
    cdef readonly double max_battery_cap
    cdef readonly double min_charging_power
    cdef readonly double max_charging_power
    cdef readonly int n_charging_points
    cdef readonly double max_phase_difference_violations
    cdef readonly double max_grid_power_violations
    cdef readonly double REWARD_SCALING
    cdef readonly double PENALTY_SCALING
    cdef readonly double BASE_PENALTY

    cdef ChargingPark charging_park
    cdef int step_counter
    cdef list supply
    cdef list soc_list
    cdef list pv_profile
    cdef double pv_power
    cdef list load_profile
    cdef int car_counter
    cdef double cumulative_soc_reward
    cdef double l1_l2
    cdef double l1_l3
    cdef double l2_l3
    cdef double ext_grid

    cdef void __reset_state(self)
    cdef double __decode_charging_phases(self, list connected_phases)
    cdef double __normalize(self, double value, double value_min, double value_max) nogil
    cdef double __clip(self, double value, double min_value, double max_value) nogil
    cdef list __get_obs(self, double pv_power, int mod, int step)
    cdef double __calculate_penalty(self, double phase_difference_violations, double grid_power_violations)
    cdef double __load_reward(self, list load_profile)
    cdef double __pv_reward(self, list pv_profile, list load_profile)
    cpdef void __track_soc_rewards(self, dict departures)
    cpdef double __soc_reward(self)
    cdef tuple __get_supply(self, int time_step, int env_id)
    cdef dict __get_arrivals(self, int time_step)
    cdef dict __get_departures(self, int time_step)
    cdef tuple __check_constraints(self)
    cdef tuple __simulation_step(self, int step_counter)
    cpdef tuple reset(self)
    cpdef tuple step(self, list action)

