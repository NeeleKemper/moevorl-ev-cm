from gymnasium.envs.registration import register

from simulation.ScenarioLoader import ScenarioLoader

register(
    id='ev-charging-test-v0',
    entry_point='envs.ev_charging_test.ev_charging_test:EVChargingTest',
    nondeterministic=True,
    kwargs={'scenario_loader': ScenarioLoader, 'env_id': int}
)
