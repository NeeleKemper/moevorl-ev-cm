import numpy as np
from typing import List, Tuple
from simulation.ChargingPark import ChargingPark


def calculate_phase_difference(l1: float, l2: float, l3: float) -> Tuple[float, float, float, float]:
    """Calculate phase differences and return the maximum difference."""
    l1_l2 = round(abs(l1 - l2), 6)
    l1_l3 = round(abs(l1 - l3), 6)
    l2_l3 = round(abs(l2 - l3), 6)

    max_diff = max(l1_l2, l1_l3, l2_l3)
    return l1_l2, l1_l3, l2_l3, max_diff


class LoadManagement:
    def __init__(self, charging_park: ChargingPark, max_grid_node_power: float, max_phase_difference: float,
                 current: float = 16):
        self.charging_park = charging_park
        self.max_grid_node_power = max_grid_node_power
        self.max_phase_difference = max_phase_difference
        self.current = current
        self.prio_charging_stations = [0] * self.charging_park.n_charging_points
        self.counter = 100000

    def calculate_action(self) -> List[float]:
        """Calculate and return the actions to be taken by charging stations."""
        actions = [0] * self.charging_park.n_charging_points
        grid_current = [0] * 3  # Representing L1, L2, L3 currents
        grid_node_power = 0

        # Update priorities based on charging points' states
        for i, cp in enumerate(self.charging_park.charging_points):
            if cp.state in ['A', 'B'] and self.prio_charging_stations[i] != 0:
                self.prio_charging_stations[i] = 0  # Reset on departure
            elif cp.state == 'C' and self.prio_charging_stations[i] == 0:
                self.counter -= 1
                self.prio_charging_stations[i] = self.counter  # Set priority on arrival

        # Sort charging points by priority (higher priority first)
        sorted_indexes = sorted(
            range(len(self.prio_charging_stations)),
            key=lambda x: self.prio_charging_stations[x],
            reverse=True
        )

        # Process charging based on priority
        for i in sorted_indexes:
            if self.prio_charging_stations[i] == 0:
                continue

            ev = self.charging_park.charging_points[i].ev
            if ev.n_charging_phases == 3:
                power = self.current * (3 ** 0.5) * 400
                if grid_node_power + power <= self.max_grid_node_power:
                    grid_current[0] += self.current
                    grid_current[1] += self.current
                    grid_current[2] += self.current
                    grid_node_power += power
                    actions[i] = 1

            elif ev.n_charging_phases == 2:
                power = self.current * 2 * 230
                l1_l2, l1_l3, l2_l3, max_diff = calculate_phase_difference(
                    grid_current[0] + self.current, grid_current[1] + self.current, grid_current[2]
                )

                if grid_node_power + power <= self.max_grid_node_power:
                    if max_diff <= self.max_phase_difference:
                        grid_current[0] += self.current
                        grid_current[1] += self.current
                        grid_node_power += power
                        actions[i] = 1
                    else:
                        regulated_current = max(self.current - (max_diff - self.max_phase_difference), 0)
                        grid_current[0] += regulated_current
                        grid_current[1] += regulated_current
                        grid_node_power += power
                        actions[i] = regulated_current / self.current

            else:  # Single-phase charging
                power = self.current * 230
                l1_l2, l1_l3, l2_l3, max_diff = calculate_phase_difference(
                    grid_current[0] + self.current, grid_current[1], grid_current[2]
                )

                if grid_node_power + power <= self.max_grid_node_power:
                    if max_diff <= self.max_phase_difference:
                        grid_current[0] += self.current
                        actions[i] = 1
                    else:
                        regulated_current = max(self.current - (max_diff - self.max_phase_difference), 0)
                        grid_current[0] += regulated_current
                        actions[i] = regulated_current / self.current

        return actions
