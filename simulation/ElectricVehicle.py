class ElectricVehicle(object):
    def __init__(self, model: str, start_soc: float, start_charging_time: int, battery_capacity_kwh: float,
                 charging_power: float, n_charging_phases: int):
        """
        Constructor for the ElectricVehicle class.

        :param model: The model of the electric vehicle.
        :param start_soc: The starting state of charge of the vehicle's battery.
        :param start_charging_time: The starting loading time of the vehicle's battery.
        :param battery_capacity_kwh: The capacity of the vehicle's battery in kilowatt-hours.
        :param charging_power: The charging power of the vehicle's battery in kilowatts.
        :param n_charging_phases: The number of load phases for the vehicle's charging process.
        """
        self.model = model
        self.battery_capacity = battery_capacity_kwh * 1e3  # kilo watt into watt
        self.max_charging_power = charging_power * 1e3  # kilo watt into watt
        self.n_charging_phases = n_charging_phases
        self.soc = start_soc
        self.charging_time = start_charging_time

    def calculate_soc(self, charging_power: float, dt: int = 1) -> float:
        """
         Calculates the state of charge (SoC) of the vehicle's battery based on the power and time interval.
        :param charging_power: The power flowing into the battery.
        :param dt: The time interval in minutes. Default is 1 minutes.
        :return: The updated state of charge of the vehicle's battery.
        """
        # Methode: Coulomb Counting
        q = charging_power * (dt / 60)
        soc = self.soc + (q / self.battery_capacity)
        return soc

    def update_charging_info(self, new_soc: float) -> None:
        """
        Reduces the charging time of the vehicle's battery by 1 and update the SoC.
        :param new_soc: The SoC after a charging cycle
        """
        self.charging_time -= 1
        self.soc = new_soc
