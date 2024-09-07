# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

cdef class ElectricVehicle:
    @classmethod
    def create(cls, unicode model, double start_soc, int start_charging_time,
               double battery_capacity_kwh, double charging_power, int n_charging_phases):
        cdef ElectricVehicle ev = ElectricVehicle()
        ev._init(model, start_soc, start_charging_time, battery_capacity_kwh, charging_power, n_charging_phases)
        return ev

    cdef void _init(self, unicode model, double start_soc, int start_charging_time,
                    double battery_capacity_kwh, double charging_power, int n_charging_phases):
        self.model = model
        self.battery_capacity = battery_capacity_kwh * 1000.0  # kilo watt into watt
        self.max_charging_power = charging_power * 1000.0  # kilo watt into watt
        self.n_charging_phases = n_charging_phases
        self.soc = start_soc
        self.charging_time = start_charging_time
    cdef double calculate_soc(self, double charging_power, double dt):
        cdef double q = charging_power * (dt / 60.0)
        cdef double soc = self.soc + (q / self.battery_capacity)
        return soc

    cdef void update_charging_info(self, double new_soc):
        self.charging_time -= 1
        self.soc = new_soc
