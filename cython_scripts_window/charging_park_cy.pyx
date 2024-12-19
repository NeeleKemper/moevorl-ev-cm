# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False,  language_level=3
# noinspection PyUnresolvedReferences
from cython_scripts.electric_vehicle_cy cimport ElectricVehicle
# noinspection PyUnresolvedReferences
from cython_scripts.charging_point_cy cimport ChargingPoint, ReturnValue

cdef class ChargingPark:
    def __init__(self, int n_charging_points, bint alternating_phase_connection, int max_charging_point_current,
                 dict ev_data_dict):
        self.n_charging_points = n_charging_points
        self.ev_json = ev_data_dict
        self.alternating_phase_connection = alternating_phase_connection
        self.max_charging_point_current = max_charging_point_current
        self.charging_points = self.__init_charging_points(n_charging_points, alternating_phase_connection)
        self.power = 0
        self.i_a, self.i_b, self.i_c = 0, 0, 0

    cdef list __init_charging_points(self, int n_charging_points, bint alternating_connection):
        cdef list charging_points = []
        cdef int n, phase_wiring
        cdef unicode cp_id
        for n in range(n_charging_points):
            if alternating_connection:
                if n % 3 == 0:
                    phase_wiring = 0
                elif n % 3 == 1:
                    phase_wiring = 1
                else:
                    phase_wiring = 2
            else:
                phase_wiring = 0
            cp_id = f'cp_{n + 1}'
            charging_points.append(ChargingPoint.create(cp_id, phase_wiring))
        return charging_points

    cdef void reset(self):
        self.charging_points = self.__init_charging_points(self.n_charging_points, self.alternating_phase_connection)
        self.power = 0
        self.i_a, self.i_b, self.i_c = 0, 0, 0

    cdef void assign_arrivals(self, dict arrivals):
        cdef int n, i
        n = arrivals['length']
        for i in range(n):
            cp = self.charging_points[arrivals['station'][i]]
            car = self.ev_json[arrivals['car'][i]]
            ev = ElectricVehicle.create(arrivals['car'][i], arrivals['soc'][i], arrivals['charging_time'][i],
                                        car['battery_capacity_kwh'],
                                        car[f'charging_power_{self.max_charging_point_current}A'],
                                        car['n_charging_phases'])
            self.__connect_ev_to_cp(cp, ev)
            self.__start_charging(cp)

    cdef void remove_departures(self, dict departures):
        n = departures['length']
        for i in range(n):
            cp = self.charging_points[departures['station'][i]]
            self.__disconnect_ev_from_cp(cp)

    cdef tuple[double, double, double, double] evaluate_cycle(self, list actions):
        cdef double total_power, total_i_a, total_i_b, total_i_c
        cdef double cm_current
        cdef ChargingPoint cp
        cdef int n
        total_power, total_i_a, total_i_b, total_i_c = 0.0, 0.0, 0.0, 0.0
        for n in range(self.n_charging_points):
            cp = self.charging_points[n]
            if self.is_cp_active(cp):
                cm_current = actions[n]
                _, _, p, _, _, _, i_a, i_b, i_c = self.__charge_ev(cp, cm_current)
                total_power += p
                total_i_a += i_a
                total_i_b += i_b
                total_i_c += i_c
        return total_power, total_i_a, total_i_b, total_i_c

    cdef void charging_cycle(self, list actions):
        cdef double total_power, total_i_a, total_i_b, total_i_c
        cdef double cm_current
        cdef ChargingPoint cp
        cdef int n

        total_power, total_i_a, total_i_b, total_i_c = 0.0, 0.0, 0.0, 0.0

        for n in range(self.n_charging_points):
            cp = self.charging_points[n]
            cm_current = actions[n]

            if self.is_cp_active(cp):
                new_soc, charging_current, p, _, _, _, i_a, i_b, i_c = self.__charge_ev(cp, cm_current)
                self.__update_ev_info(cp, new_soc)
            else:
                p, i_a, i_b, i_c = 0.0, 0.0, 0.0, 0.0

            total_power += p
            total_i_a += i_a
            total_i_b += i_b
            total_i_c += i_c

        self.power = total_power
        self.i_a = total_i_a
        self.i_b = total_i_b
        self.i_c = total_i_c

    cdef tuple[double, double, double] get_i(self):
        return self.i_a, self.i_b, self.i_c

    cdef double get_power(self):
        return self.power

    cdef list get_charging_points(self):
        return self.charging_points

    cdef bint is_cp_active(self, ChargingPoint charging_point):
        return charging_point.is_active()

    cdef ElectricVehicle get_ev_properties_of_cp(self, ChargingPoint charging_point):
        if self.is_cp_active(charging_point):
            return charging_point.ev
        return None

    cdef list get_cp_connected_phases(self, ChargingPoint charging_point):
        return charging_point.connected_phases

    cdef inline void __connect_ev_to_cp(self, ChargingPoint charging_point, ElectricVehicle electric_vehicle):
        charging_point.connect(electric_vehicle)

    cdef inline void __disconnect_ev_from_cp(self, ChargingPoint charging_point):
        charging_point.disconnect()

    cdef inline void __start_charging(self, ChargingPoint charging_point):
        charging_point.start_charging()

    cdef inline ReturnValue __charge_ev(self, ChargingPoint charging_point, double current):
        return charging_point.charge(current)

    cdef inline void __update_ev_info(self, ChargingPoint charging_point, double new_soc):
        charging_point.update_charging_info(new_soc)
