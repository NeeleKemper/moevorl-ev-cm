# electric_vehicle.pxd
cdef class ElectricVehicle:
    cdef readonly unicode model
    cdef readonly double battery_capacity
    cdef readonly double max_charging_power
    cdef readonly int n_charging_phases
    cdef double soc
    cdef int charging_time

    cdef void _init(self, unicode model, double start_soc, int start_charging_time,
                    double battery_capacity_kwh, double charging_power, int n_charging_phases)
    cdef double calculate_soc(self, double charging_power, double dt)
    cdef void update_charging_info(self, double new_soc)
