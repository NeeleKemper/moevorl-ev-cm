from typing import Tuple
from simulation.ElectricVehicle import ElectricVehicle
from simulation.ChargingPoint import ChargingPoint


class ChargingPark(object):
    def __init__(self, n_charging_points: int, alternating_phase_connection: bool, max_charging_point_current: int,
                 ev_data_dict: dict) -> None:
        """
        Initializes the ChargingPark object with a given number of charging points, electric vehicles data
        and other related dataframes for power and current values.

        :param n_charging_points: The number of charging points in the park.
        """
        self.n_charging_points = n_charging_points
        self.ev_json = ev_data_dict
        self.alternating_phase_connection = alternating_phase_connection
        self.max_charging_point_current = max_charging_point_current
        self.charging_points = self.__init_charging_points(n_charging_points, alternating_phase_connection)

        self.power = 0
        self.i_a, self.i_b, self.i_c = 0, 0, 0

    @staticmethod
    def __init_charging_points(n_charging_points: int, alternating_connection) -> list:
        """
        Initializes the charging points in the park.

        :param n_charging_points: The number of charging points to be created.
        :return: A list containing all charging point objects.
        """
        charging_points = []
        for n in range(n_charging_points):
            if alternating_connection:
                if n % 3 == 0:
                    phase_wiring = 'l1-l2-l3'
                elif n % 3 == 1:
                    phase_wiring = 'l2-l3-l1'
                else:  # i.e., when n%3 == 2
                    phase_wiring = 'l3-l1-l2'
            else:
                phase_wiring = 'l1-l2-l3'
            cp_id = f'cp_{n + 1}'
            cp = ChargingPoint(cp_id, phase_wiring)
            charging_points.append(cp)
        return charging_points

    def reset(self):
        """
        Resets the internal state of the charging park.
        :return: None
        """
        self.charging_points = self.__init_charging_points(self.n_charging_points, self.alternating_phase_connection)
        self.power = 0
        self.i_a, self.i_b, self.i_c = 0, 0, 0

    def assign_arrivals(self, arrivals: dict):
        """
        Assigns the arriving electric vehicles to the charging points.

        :param arrivals: A dictionary containing the arriving electric vehicles' data.
        """
        n = arrivals['length']
        evs, cps=[], []
        for i in range(n):
            cp = self.charging_points[arrivals['station'][i]]
            cps.append(arrivals['station'][i])
            car = self.ev_json[arrivals['car'][i]]
            ev = ElectricVehicle(arrivals['car'][i], arrivals['soc'][i], arrivals['charging_time'][i],
                                 car['battery_capacity_kwh'], car[f'charging_power_{self.max_charging_point_current}A'],
                                 car['n_charging_phases'])
            evs.append(ev)
            cp.connect(ev)
            cp.start_charging()

    def remove_departures(self, departures: dict):
        """
        Disconnects the departing electric vehicles from the charging points.

        :param departures: A DataFrame containing the departing electric vehicles' data.
        """
        n = departures['length']
        for i in range(n):
            cp = self.charging_points[departures['station'][i]]
            cp.disconnect()

    def evaluate_cycle(self, actions: list) -> Tuple[float, float, float, float]:
        total_power, total_i_a, total_i_b, total_i_c = 0.0, 0.0, 0.0, 0.0
        for cp, cm_current in zip(self.charging_points, actions):
            if cp.is_active():
                _, _, p, _, _, _, i_a, i_b, i_c = cp.evaluate(cm_current)
                total_power += p
                total_i_a += i_a
                total_i_b += i_b
                total_i_c += i_c
        return total_power, total_i_a, total_i_b, total_i_c

    def charging_cycle(self, actions: list) -> None:
        """
        Executes a loading cycle for each active charging point in the park.

        :param actions: The assigned current for each charging point
        :param update:
        """
        total_power, total_i_a, total_i_b, total_i_c = 0.0, 0.0, 0.0, 0.0
        for cp, cm_current in zip(self.charging_points, actions):
            if cp.is_active():
                new_soc, charging_current, p, _, _, _, i_a, i_b, i_c = cp.charge(cm_current)
                cp.update_charging_info(new_soc)
            else:
                _, charging_current, p, _, _, _, i_a, i_b, i_c = 0, 0, 0, 0, 0, 0, 0, 0, 0
            total_power += p
            total_i_a += i_a
            total_i_b += i_b
            total_i_c += i_c
        self.power = total_power
        self.i_a = total_i_a
        self.i_b = total_i_b
        self.i_c = total_i_c

    def get_i(self) -> list:
        return [self.i_a, self.i_b, self.i_c]

    def get_power(self) -> float:
        return self.power

    def get_charging_points(self) -> list:
        return self.charging_points

    @staticmethod
    def is_cp_active(charging_point: ChargingPoint) -> bool:
        return charging_point.is_active()

    def get_ev_properties_of_cp(self, charging_point: ChargingPoint) -> ElectricVehicle:
        if self.is_cp_active(charging_point):
            ev = charging_point.ev
            return ev
        return None

    @staticmethod
    def get_cp_connected_phases(charging_point: ChargingPoint) -> list:
        return charging_point.connected_phases
