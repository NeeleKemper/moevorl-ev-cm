import numpy as np
import gymnasium as gym
from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.spaces import Box

from cython_scripts.environment_cy import Environment
from simulation.ScenarioLoader import ScenarioLoader
from simulation.constant import (MAX_PV_POWER, MAX_CHARGING_TIME, MIN_BATTERY_CAP, MAX_BATTERY_CAP, MIN_CHARGING_POWER,
                                 MAX_CHARGING_POWER)

REWARD_SCALING = 100
PENALTY_SCALING = 1
BASE_PENALTY = -1


class EVCharging(gym.Env):
    def __init__(self, scenario_loader: ScenarioLoader,  env_id=-1):
        self.id = env_id
        # Initialize variables
        # Generate scenario and unpack variables
        supply, event_generator, charging_park, time_steps = scenario_loader.generate_scenario(env_id)

        # Retrieve necessary values from scenario
        scenario = scenario_loader.scenario

        n_charging_points = scenario['n_charging_points']
        action_scaler = scenario['maximum_charging_point_current']
        max_phase_difference = scenario['maximum_phase_difference']
        min_required_soc = scenario['minimum_required_soc']

        # Calculate power values
        max_grid_node_power = scenario['maximum_grid_node_power'] * 1e3
        max_pv_power = np.ceil(MAX_PV_POWER * scenario['pv_scaling'])
        max_power = max_pv_power + max_grid_node_power
        max_current = max_power / ((3 ** 0.5) * 400)

        # Set state dimensions
        n_local_states = 6
        n_global_states = 10
        n_cp_state = n_charging_points * n_local_states
        n_states = n_global_states + n_cp_state
        self.observation_space = Box(low=0, high=1, shape=(n_states,), dtype=np.float64)

        # Set continues action space
        self.action_space = Box(low=0, high=1.0, shape=(n_charging_points,), dtype=np.float64)

        # Set reward space
        self.reward_dim = 3
        self.reward_space = Box(
                low=np.array(
                    [BASE_PENALTY - PENALTY_SCALING, BASE_PENALTY - PENALTY_SCALING, BASE_PENALTY - PENALTY_SCALING]),
                high=np.array([REWARD_SCALING, REWARD_SCALING, REWARD_SCALING]),
                shape=(self.reward_dim,),
                dtype=np.float64
            )

        if env_id == -1:
            power_supply = supply.supply
        else:
            power_supply = supply.supply_list

        self.environment = Environment(charging_park=charging_park, supply=power_supply,
                                       departures=dict(event_generator.departures),
                                       arrivals=dict(event_generator.arrivals), env_id=env_id,
                                       action_scaler=action_scaler, time_steps=time_steps,
                                       min_required_soc=min_required_soc,
                                       max_power=max_power, max_pv_power=max_pv_power,
                                       max_grid_node_power=max_grid_node_power, max_current=max_current,
                                       max_charging_time=MAX_CHARGING_TIME, max_phase_difference=max_phase_difference,
                                       n_charging_points=n_charging_points, min_battery_cap=MIN_BATTERY_CAP-100,
                                       max_battery_cap=MAX_BATTERY_CAP, min_charging_power=MIN_CHARGING_POWER-100,
                                       max_charging_power=MAX_CHARGING_POWER)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        obs, _ = self.environment.reset()
        return np.array(obs), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        next_obs, vec_reward, terminal, _, _ = self.environment.step(action)
        vec_reward = np.array(vec_reward, dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
        return next_obs, vec_reward, terminal, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def pareto_front(self, gamma: float):
        pass
