import json
from jsonschema import validate, ValidationError
from typing import Tuple
from simulation.Supply import Supply

from simulation.ChargingEventGenerator import ChargingEventGenerator

# A schema that describes the expected structure of a scenario configuration object.
SCHEMA = {
    'type': 'object',
    'properties': {
        'n_charging_points': {'type': 'integer', 'minimum': 1},
        'pv_scaling': {'type': 'number', 'minimum': 0, 'exclusiveMinimum': 0},
        'alternating_phase_connection': {'type': 'boolean'},
        'charging_event_density': {'type': 'number', 'minimum': 0, 'exclusiveMinimum': 0},
        'maximum_grid_node_power': {'type': 'number', 'minimum': 0},
        'maximum_charging_point_current': {'enum': [16, 32]},
        'maximum_phase_difference': {'type': 'number', 'minimum': 16, 'maximum': 200},
        'minimum_required_soc': {'type': 'number', 'minimum': 0, 'maximum': 1, 'exclusiveMinimum': 0},
    },
    'required': ['n_charging_points', 'pv_scaling', 'charging_event_density', 'alternating_phase_connection',
                 'maximum_grid_node_power', 'maximum_charging_point_current', 'maximum_phase_difference',
                 'minimum_required_soc']
}


class ScenarioLoader(object):
    def __init__(self, scenario_name: str, data_set: str, charging_density: str = 'norm',
                 pv_noise: bool = False, seed: int = 42):
        """
        Initialize the ScenarioLoader class. Loads a scenario from a json file based on a provided scenario name.
        """
        self.seed = seed
        self.pv_noise = pv_noise
        try:
            with open('scenarios.json') as f:
                scenarios = json.load(f)
                self.scenario = scenarios[scenario_name]
        except FileNotFoundError:
            raise FileNotFoundError('scenarios.json not found.')
        self.__check_scenario_config(self.scenario)


        self.event_generator = ChargingEventGenerator(n_charging_points=self.scenario['n_charging_points'],
                                                      max_charging_point_current=self.scenario[
                                                          'maximum_charging_point_current'],
                                                      density=self.scenario['charging_event_density'],
                                                      density_note=charging_density,
                                                      minimum_required_soc=self.scenario['minimum_required_soc'],
                                                      seed=self.seed,
                                                      scenario=scenario_name)

        self.supply = Supply(pv_scaling=self.scenario['pv_scaling'],
                             pv_noise= self.pv_noise,
                             seed=self.seed)
        ev_json = json.load(open('electric_vehicles.json'))
        if data_set == 'test' or data_set == 'all':
            from simulation.ChargingPark import ChargingPark
        else:
            from cython_scripts.charging_park_cy import ChargingPark

        self.charging_park = ChargingPark(n_charging_points=self.scenario['n_charging_points'],
                                          alternating_phase_connection=self.scenario[
                                              'alternating_phase_connection'],
                                          max_charging_point_current=self.scenario[
                                              'maximum_charging_point_current'],
                                          ev_data_dict=ev_json)

        self.n_envs = self.supply.split_supply(self.event_generator.schedule, data_set)
        print(f'Number of envs: {self.n_envs}')

    @staticmethod
    def __check_scenario_config(scenario: dict) -> None:
        """
        Check the validity of a scenario configuration against the expected schema.

        :param scenario: dict, The scenario configuration to validate.
        :return: None
        Raises ValueError if the configuration is not valid.
        """
        try:
            validate(instance=scenario, schema=SCHEMA)
        except ValidationError as e:
            raise ValueError(f'Invalid scenario config: {e.message}')

    def generate_scenario(self, env_id: int) -> Tuple[Supply, ChargingEventGenerator, any, list]:
        """
        Generate a scenario based on the loaded scenario configuration.

        :param env_id: int
        :return: Tuple[Supply, ChargingEventGenerator, ChargingPark, ChargingManagement, list],
        Returns a tuple containing an instance of PVSupply, ChargingEventGenerator, ChargingPark, ChargingManagement
        and a list of time steps for the scenario.
        """

        # Use state importance sampling instead of filtering out irrelevant states.
        # This allows the model to learn primarily from the most informative states while still preserving the full
        # sequence of states.

        if env_id == -1:
            time_steps = self.supply.supply[:, 0].tolist()
        else:
            time_steps = self.supply.supply_list[env_id][:, 0].tolist()

        return self.supply, self.event_generator, self.charging_park, time_steps
