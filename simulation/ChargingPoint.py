from typing import Tuple

from simulation.ElectricVehicle import ElectricVehicle
from simulation.constant import SOC_OPT_THRESHOLD, ETA

VOLTAGE_PER_PHASE = 230


class ChargingPoint(object):
    def __init__(self, cp_id: str, phase_wiring: str):
        """
        Constructor for the ChargingPoint class.

        :param cp_id: The unique identifier of the charging point.
        :param phase_wiring: A string representing the wiring configuration of the charging point.
        """
        self.id = cp_id
        self.phase_wiring = phase_wiring
        self.ev = None

        # state: A -> free; B -> connected; C -> charging
        self.state = 'A'

        self.connected_phases = [0, 0, 0]
        self.charging_voltage_phase = 0

    def __connect_phases(self, n_charging_phases: int) -> list:
        """
        Returns a list indicating the connected phases based on the phase wiring and the number of load phases.

        :param n_charging_phases: The number of load phases of the electric vehicle.
        :return: A list indicating the connected phases.
        """
        phase_mapping = {
            'l1-l2-l3': [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            'l3-l1-l2': [[0, 1, 0], [0, 1, 1], [1, 1, 1]],
            'l2-l3-l1': [[0, 0, 1], [1, 0, 1], [1, 1, 1]],
        }

        return phase_mapping[self.phase_wiring][n_charging_phases - 1]

    def __charging_reduction(self, soc: float) -> float:
        """
        Returns the reduced charging power based on the state of charge (soc).

        :param soc: The current state of charge of the electric vehicle.
        :return: The reduced charging power.
        """
        m = self.ev.max_charging_power / (1 - SOC_OPT_THRESHOLD)
        power = -m * soc + m
        return power

    def __calculate_charging_power(self, soc: float, max_charging_power: float) -> float:
        """
        Calculates the charging power based on the state of charge and the maximum charging power.

        :param soc: The current state of charge of the electric vehicle.
        :param max_charging_power: The maximum charging power that can be provided by the charging point.
        :return: The actual charging power.
        """
        if soc <= SOC_OPT_THRESHOLD:
            charging_power = max_charging_power
        else:
            charging_power = min(self.__charging_reduction(soc), max_charging_power)

        charging_power *= ETA  # Power actually delivered to the battery

        return charging_power

    def __calculate_current(self, charging_power: float) -> float:
        """
        Calculates the current based on the charging power.

        :param charging_power: The charging power of the electric vehicle.
        :return: The current required to deliver the charging power.
        """
        return charging_power / self.charging_voltage_phase

    def connect(self, ev: ElectricVehicle) -> None:
        """
        Connects an electric vehicle to the charging point.

        :param ev: The ElectricVehicle instance to be connected to the charging point.
        """
        self.ev = ev
        self.state = 'B'
        self.connected_phases = self.__connect_phases(ev.n_charging_phases)

        # set charging voltage and max current charging voltage values depending on the number of load phases

        # Charging power(single - phase alternating current):
        # Charging power(3.7 kW) = phases(1) * voltage(230 V) * amperage(16 A).

        # Charging power(three - phase alternating current), star connection:
        # Charging power(22 kW) = phases(3) * voltage(230 V) * amperage(32 A)

        # Alternative: Charging power(three - phase current, three - phase alternating current), deltaconnection:
        # Charging power(22 kW) = root(3) * voltage(400 V) * amperage(32 A)

        if ev.n_charging_phases == 1:
            self.charging_voltage_phase = VOLTAGE_PER_PHASE
        elif ev.n_charging_phases == 2:
            self.charging_voltage_phase = 2 * VOLTAGE_PER_PHASE
        else:
            # sqrt(3) * 400 = 3 * 230
            self.charging_voltage_phase = (3 ** 0.5) * 400

    def charge(self, cm_current: float) -> Tuple[
        float, float, float, float, float, float, float, float, float]:
        """
        Executes a charging cycle and returns the charging power for each phase.

        :param cm_current: The loading current provided by the charging management
        :param evaluate:
        :return: A tuple representing the new state of charge, the used charging current,
        the charging power for each phase and the current for each phase.
        """
        max_charging_power = min(self.ev.max_charging_power, cm_current * self.charging_voltage_phase)
        charging_power = self.__calculate_charging_power(self.ev.soc, max_charging_power)
        charging_current = self.__calculate_current(charging_power)
        new_soc = self.ev.calculate_soc(charging_power, dt=1)

        # calculate charging power for each phase
        power_factor = 1 / int(self.ev.n_charging_phases)
        # divide by eta to get the actual power drawn from the grid
        p_a = self.connected_phases[0] * power_factor * charging_power / ETA
        p_b = self.connected_phases[1] * power_factor * charging_power / ETA
        p_c = self.connected_phases[2] * power_factor * charging_power / ETA
        p = p_a + p_b + p_c

        i_a, i_b, i_c = p_a / 230, p_b / 230, p_c / 230

        return new_soc, charging_current, p, p_a, p_b, p_c, i_a, i_b, i_c

    def evaluate(self, cm_current: float) -> Tuple[
        float, float, float, float, float, float, float, float, float]:
        """
        Executes a charging cycle and returns the charging power for each phase.

        :param cm_current: The loading current provided by the charging management
        :param evaluate:
        :return: A tuple representing the new state of charge, the used charging current,
        the charging power for each phase and the current for each phase.
        """
        max_charging_power = min(self.ev.max_charging_power, cm_current * self.charging_voltage_phase)
        charging_power = self.__calculate_charging_power(self.ev.soc, max_charging_power)

        # calculate charging power for each phase
        power_factor = 1 / int(self.ev.n_charging_phases)
        # divide by eta to get the actual power drawn from the grid
        p_a = self.connected_phases[0] * power_factor * charging_power / ETA
        p_b = self.connected_phases[1] * power_factor * charging_power / ETA
        p_c = self.connected_phases[2] * power_factor * charging_power / ETA
        p = p_a + p_b + p_c

        i_a, i_b, i_c = p_a / 230, p_b / 230, p_c / 230
        return 0, 0, p, p_a, p_b, p_c, i_a, i_b, i_c

    def disconnect(self) -> None:
        """
        Disconnects the current electric vehicle from the charging point.
        """
        self.ev = None
        self.state = 'A'
        self.charging_voltage_phase = 0
        self.connected_phases = [0, 0, 0]

    def start_charging(self) -> None:
        """
        Sets the charging point state to "charging".
        """
        self.state = 'C'

    def is_active(self):
        """
        Returns whether the charging point is active (i.e., currently connected to an electric vehicle).

        :return: Boolean indicating whether the charging point is active.
        """
        return self.state != 'A' and self.ev is not None

    def update_charging_info(self, new_soc: float) -> None:
        if new_soc >= 0.99:
            new_soc = 1
            self.state = 'B'
        else:
            self.state = 'C'
        self.ev.update_charging_info(new_soc)
