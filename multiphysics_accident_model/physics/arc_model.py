import numpy as np


class ArcModel:
    def __init__(self, supply_mode):
        self.supply_mode = supply_mode

    @property
    def supply_mode(self):
        return self._supply_mode

    @supply_mode.setter
    def supply_mode(self, value):
        value = str(value).upper()
        if value not in {'DC', 'AC'}:
            raise ValueError("supply_mode must be 'DC' or 'AC'")
        self._supply_mode = value


def set_arc_output(sim, voltage, current, power, source_wave):
    sim.arc_on = bool(power > 1e-6 and current > 1e-6)
    sim.arc_voltage = float(voltage)
    sim.arc_current = float(current)
    sim.arc_power = float(power)
    return voltage, current, power, source_wave


def get_source_wave(sim, curr_t):
    if sim.arc_model.supply_mode == 'DC':
        return sim.source_voltage_nominal
    return np.sqrt(2.0) * sim.source_voltage_nominal * np.sin(2.0 * np.pi * sim.grid_freq_hz * curr_t + sim.ac_source_phase)


def compute_dc_arc_branch(sim, series_resistance, intermittent):
    gap_mm = max(sim.arc_gap_mm, 0.05 if intermittent else 0.12)
    i_est = max((sim.source_voltage_nominal - sim.arc_voltage_base) / max(series_resistance, 1e-6), 0.0)
    v_arc = sim.arc_voltage_base + sim.arc_field_v_per_mm * gap_mm + sim.k_arc_ayrton / max(i_est, sim.min_arc_hold_current)
    v_arc += sim.rng.normal(0.0, 1.0 if intermittent else 0.45)
    i_line = max(0.0, (sim.source_voltage_nominal - v_arc) / max(series_resistance, 1e-6))
    duty = 1.0
    if intermittent:
        duty = float(np.clip(0.10 + 0.78 * sim.carb_track[sim.f_node] + sim.rng.normal(0.0, 0.04), 0.05, 0.90))
        i_line *= duty
    p_arc = max(0.0, v_arc * i_line)
    return abs(v_arc), i_line, p_arc, duty, i_line, v_arc, sim.source_voltage_nominal


def compute_ac_arc_branch(sim, series_resistance, intermittent, curr_t):
    source_wave = sim._get_source_wave(curr_t)
    source_mag = abs(source_wave)
    source_sign = 1 if source_wave >= 0.0 else -1
    phase_mag = abs(np.sin(2.0 * np.pi * sim.grid_freq_hz * curr_t + sim.ac_source_phase))

    if source_sign != sim.prev_ac_source_sign:
        sim.ac_zero_cross_total += 1
        sim.prev_ac_source_sign = source_sign
        sim.ac_arc_conducting = False

    if source_mag < sim.ac_extinction_voltage or phase_mag < sim.ac_phase_extinction_band:
        sim.ac_arc_conducting = False
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, source_wave

    gap_mm = max(sim.arc_gap_mm, 0.06 if intermittent else 0.14)
    restrike_threshold = sim.ac_restrike_voltage + 2.6 * gap_mm
    reignition_boost = 0.0
    if not sim.ac_arc_conducting:
        if source_mag < restrike_threshold:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, source_wave
        sim.ac_arc_conducting = True
        sim.ac_reignition_total += 1
        reignition_boost = sim.ac_reignition_overshoot

    i_est = max((source_mag - sim.arc_voltage_base) / max(series_resistance, 1e-6), 0.0)
    v_arc_mag = sim.arc_voltage_base + sim.arc_field_v_per_mm * gap_mm + reignition_boost + sim.k_arc_ayrton / max(i_est, sim.min_arc_hold_current)
    v_arc_mag += sim.rng.normal(0.0, 1.4 if intermittent else 0.8)
    i_mag = max(0.0, (source_mag - v_arc_mag) / max(series_resistance, 1e-6))
    duty = 1.0
    if intermittent:
        duty = float(np.clip(0.06 + 0.58 * sim.carb_track[sim.f_node] + 0.18 * phase_mag + sim.rng.normal(0.0, 0.05), 0.04, 0.78))
        i_mag *= duty
    p_arc = max(0.0, v_arc_mag * i_mag)
    i_wave = source_sign * i_mag
    v_arc_wave = source_sign * v_arc_mag
    return v_arc_mag, i_mag, p_arc, duty, i_wave, v_arc_wave, source_wave


def compute_arc_branch(sim, series_resistance, intermittent, curr_t):
    if sim.arc_model.supply_mode == 'DC':
        result = sim._compute_dc_arc_branch(series_resistance, intermittent)
    else:
        result = sim._compute_ac_arc_branch(series_resistance, intermittent, curr_t)
    voltage, current, power, duty, wave_current, fault_voltage_wave, source_voltage_wave = result
    set_arc_output(sim, voltage, current, power, source_voltage_wave)
    return voltage, current, power, duty, wave_current, fault_voltage_wave, source_voltage_wave


def required_ignition_energy(sim, arc_current, arc_power):
    current_factor = np.clip((arc_current - sim.min_ignition_current) / 27.0, 0.0, 1.0)
    required_energy = sim.arc_energy_zone_high - current_factor * (sim.arc_energy_zone_high - sim.arc_energy_zone_low)
    if arc_power >= sim.P_crit_arc_2:
        required_energy *= 0.75
    elif arc_power < sim.P_crit_arc_1:
        required_energy *= 1.15
    if sim.arc_model.supply_mode == 'DC':
        required_energy *= 0.88
    return required_energy * (1.0 - 0.10 * sim.gamma)
