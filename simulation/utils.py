import warnings
import mo_gymnasium as mo_gym
from mo_gymnasium import MORecordEpisodeStatistics
from envs.ev_charging.ev_charging import EVCharging
from envs.ev_charging_test.ev_charging_test import EVChargingTest
from simulation.ScenarioLoader import ScenarioLoader

warnings.filterwarnings('ignore')


def make_env(scenario: str = 'scenario_CS05', data_set: str = 'train', record_episode_statistics: bool = False,
             multiple_envs: bool = False, seed: int = 42) -> MORecordEpisodeStatistics | EVCharging | list[EVCharging]:
    scenario_loader = ScenarioLoader(scenario, data_set=data_set, seed=seed)
    if multiple_envs:
        env = []
        for env_id in range(scenario_loader.n_envs):
            e = mo_gym.make('envs.ev_charging:ev-charging-v0', scenario_loader=scenario_loader, env_id=env_id)
            env.append(e)
    else:
        env = mo_gym.make('envs.ev_charging:ev-charging-v0', scenario_loader=scenario_loader, env_id=-1)
        if record_episode_statistics:
            env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
    return env


def make_test_env(scenario: str = 'scenario_CS05', data_set: str = 'train', record_episode_statistics: bool = False,
                  charging_density: str = 'norm', pv_noise: bool = False, multiple_envs: bool = False, seed: int = 42) \
        -> MORecordEpisodeStatistics | EVChargingTest | list[EVChargingTest]:
    scenario_loader = ScenarioLoader(scenario, data_set=data_set, charging_density=charging_density,
                                     pv_noise=pv_noise, seed=seed)
    if multiple_envs:
        env = []
        for env_id in range(scenario_loader.n_envs):
            e = mo_gym.make('envs.ev_charging_test:ev-charging-test-v0', scenario_loader=scenario_loader,
                            env_id=env_id)
            env.append(e)
    else:
        env = mo_gym.make('envs.ev_charging_test:ev-charging-test-v0', scenario_loader=scenario_loader,
                          env_id=-1)
        if record_episode_statistics:
            env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
    return env
