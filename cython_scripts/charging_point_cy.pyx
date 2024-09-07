# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
#noinspection PyUnresolvedReferences
from cython_scripts.electric_vehicle_cy cimport ElectricVehicle

cdef class ChargingPoint:
    @classmethod
    def create(cls, unicode cp_id, int phase_wiring):
        cdef ChargingPoint cp = ChargingPoint()
        cp._init(cp_id, phase_wiring)
        return cp

    cdef void _init(self, unicode cp_id, int phase_wiring):
        self.SOC_OPT_THRESHOLD = 0.85
        self.ETA = 0.9
        self.id = cp_id
        self.phase_wiring = phase_wiring

        self.ev = None
        self.state = 1
        self.connected_phases[0] = 0
        self.connected_phases[1] = 0
        self.connected_phases[2] = 0
        self.charging_voltage_phase = 0

    cdef void __connect_phases(self, int n_charging_phases):
        cdef int[3][3] phase_mapping_L1_L2_L3 = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
        cdef int[3][3] phase_mapping_L3_L1_L2 = [
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 1]
        ]
        cdef int[3][3] phase_mapping_L2_L3_L1 = [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]

        cdef int i
        if self.phase_wiring == 0: # # l1-l2-l3
            for i in range(3):
                self.connected_phases[i] = phase_mapping_L1_L2_L3[n_charging_phases - 1][i]
        elif self.phase_wiring == 1: # l2-l3-l1
            for i in range(3):
                self.connected_phases[i] = phase_mapping_L2_L3_L1[n_charging_phases - 1][i]
        else: # l3-l1-l2:
            for i in range(3):
                self.connected_phases[i] = phase_mapping_L3_L1_L2[n_charging_phases - 1][i]
    cdef inline double __charging_reduction(self, double soc) nogil:
        cdef double m = self.ev.max_charging_power / (1.0 - self.SOC_OPT_THRESHOLD)
        cdef double power = -m * soc + m
        return power

    cdef inline double __calculate_charging_power(self, double soc, double max_charging_power) nogil:
        cdef double charging_power
        if soc <= self.SOC_OPT_THRESHOLD:
            charging_power = max_charging_power
        else:
            charging_power = min(self.__charging_reduction(soc), max_charging_power)
        charging_power *= self.ETA
        return charging_power

    cdef inline double __calculate_current(self, double charging_power) nogil:
        cdef double current = charging_power / self.charging_voltage_phase
        return current

    cdef void connect(self, ElectricVehicle ev):
        self.ev = ev
        self.state = 2
        self.__connect_phases(ev.n_charging_phases)
        if ev.n_charging_phases == 1:
            self.charging_voltage_phase = 230.0
        elif ev.n_charging_phases == 2:
            self.charging_voltage_phase = 460.0
        else:
            self.charging_voltage_phase = 3 ** 0.5 * 400.0

    cdef ReturnValue evaluate(self, double cm_current):
        cdef double max_charging_power = min(self.ev.max_charging_power, cm_current * self.charging_voltage_phase)
        cdef double charging_power = self.__calculate_charging_power(self.ev.soc, max_charging_power)
        cdef double power_factor = 1.0 / self.ev.n_charging_phases

        cdef double p[3]
        cdef double i[3]
        cdef double total_power = 0.0
        cdef int j

        for j in range(3):
            p[j] = self.connected_phases[j] * power_factor * charging_power / self.ETA
            i[j] = p[j] / 230.0
            total_power += p[j]

        return 0, 0, total_power, p[0], p[1], p[2], i[0], i[1], i[2]

    cdef ReturnValue charge(self, double cm_current):
        cdef double max_charging_power = min(self.ev.max_charging_power, cm_current * self.charging_voltage_phase)
        cdef double charging_power = self.__calculate_charging_power(self.ev.soc, max_charging_power)
        cdef double charging_current = self.__calculate_current(charging_power)
        cdef double new_soc = self.ev.calculate_soc(charging_power, dt=1.0)
        cdef double power_factor = 1.0 / self.ev.n_charging_phases
        cdef double p[3]
        cdef double i[3]
        cdef double total_power = 0
        cdef int j

        for j in range(3):
            p[j] = self.connected_phases[j] * power_factor * charging_power / self.ETA
            i[j] = p[j] / 230.0
            total_power += p[j]
        return new_soc, charging_current, total_power, p[0], p[1], p[2], i[0], i[1], i[2]

    cdef void disconnect(self):
        self.ev = None
        self.state = 1
        self.charging_voltage_phase = 0
        self.connected_phases[0] = 0
        self.connected_phases[1] = 0
        self.connected_phases[2] = 0

    cdef void start_charging(self):
        self.state = 3

    cdef bint is_active(self):
        cdef bint active = self.state != 1 and self.ev is not None
        return active

    cdef void update_charging_info(self, double new_soc):
        if new_soc >= 0.99:
            new_soc = 1
            self.state = 2
        else:
            self.state = 3
        self.ev.update_charging_info(new_soc)
