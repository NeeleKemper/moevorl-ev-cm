# charging_point.pxd
# noinspection PyUnresolvedReferences
from .electric_vehicle_cy cimport ElectricVehicle

ctypedef tuple[double, double, double, double, double, double, double, double, double] ReturnValue

cdef class ChargingPoint:
    cdef readonly unicode id
    cdef readonly int phase_wiring
    cdef readonly double SOC_OPT_THRESHOLD
    cdef readonly double ETA

    cdef ElectricVehicle ev
    cdef int state
    cdef double charging_voltage_phase
    cdef int connected_phases[3]

    cdef void _init(self, unicode cp_id, int phase_wiring)
    cdef void __connect_phases(self, int n_charging_phases)
    cdef double __charging_reduction(self, double soc) nogil
    cdef double __calculate_charging_power(self, double soc, double max_charging_power) nogil
    cdef double __calculate_current(self, double charging_power) nogil
    cdef void connect(self, ElectricVehicle ev)
    cdef ReturnValue evaluate(self, double cm_current)
    cdef ReturnValue charge(self, double cm_current)
    cdef void disconnect(self)
    cdef void start_charging(self)
    cdef bint is_active(self)
    cdef void update_charging_info(self, double new_soc)
