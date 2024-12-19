# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False,  language_level=3

cimport numpy as cnp
from .electric_vehicle_cy cimport ElectricVehicle
from .charging_point_cy cimport ChargingPoint, ReturnValue

cdef class ChargingPark:
    cdef readonly int n_charging_points
    cdef readonly int max_charging_point_current
    cdef readonly bint alternating_phase_connection

    cdef double power, i_a, i_b, i_c
    cdef list charging_points
    cdef dict ev_json

    cdef list __init_charging_points(self, int n_charging_points, bint alternating_connection)
    cdef void reset(self)
    cdef void assign_arrivals(self, dict arrivals)
    cdef void remove_departures(self, dict departures)
    cdef (double, double, double, double) evaluate_cycle(self, list actions)
    cdef void charging_cycle(self, list actions)
    cdef tuple[double, double, double] get_i(self)
    cdef double get_power(self)
    cdef list get_charging_points(self)
    cdef bint is_cp_active(self, ChargingPoint charging_point)
    cdef ElectricVehicle get_ev_properties_of_cp(self, ChargingPoint charging_point)
    cdef list get_cp_connected_phases(self, ChargingPoint charging_point)

    cdef void __connect_ev_to_cp(self, ChargingPoint charging_point, ElectricVehicle electric_vehicle)

    cdef void __disconnect_ev_from_cp(self, ChargingPoint charging_point)

    cdef void __start_charging(self, ChargingPoint charging_point)

    cdef ReturnValue __charge_ev(self, ChargingPoint charging_point, double current)

    cdef void __update_ev_info(self, ChargingPoint charging_point, double new_soc)