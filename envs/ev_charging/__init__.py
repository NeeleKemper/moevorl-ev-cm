from gymnasium.envs.registration import register

from simulation.ScenarioLoader import ScenarioLoader

register(
    id='ev-charging-v0',
    entry_point='envs.ev_charging.ev_charging:EVCharging',
    nondeterministic=True,
    kwargs={'scenario_loader': ScenarioLoader, 'env_id': int}
)
