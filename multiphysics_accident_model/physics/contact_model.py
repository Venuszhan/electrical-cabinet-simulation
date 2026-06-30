def estimate_contact_power(sim, series_resistance, fault_resistance):
    total_resistance = max(series_resistance + fault_resistance, 1e-6)
    i_equiv = sim.source_voltage_nominal / total_resistance
    return i_equiv, i_equiv ** 2 * fault_resistance


def compute_contact_branch(sim, series_resistance, fault_resistance, curr_t):
    source_wave = sim._get_source_wave(curr_t)
    total_resistance = max(series_resistance + fault_resistance, 1e-6)
    i_wave = source_wave / total_resistance
    i_abs = abs(i_wave)
    v_fault_wave = i_wave * fault_resistance
    p_fault = i_wave ** 2 * fault_resistance
    return abs(v_fault_wave), i_abs, p_fault, i_wave, v_fault_wave, source_wave


def update_fault_contact(sim, local_temp, contact_power):
    profile = sim.fault_profile
    severity = max(0.0, min(1.0, profile.severity))
    if severity <= 0.0:
        sim.R_loose = 0.0
        sim.R_oxide = 0.0
        sim.R_carbon = 0.0
        sim.states[sim.f_node][2] = sim.R_metal
        return

    temp_softening = max(0.0, local_temp - 80.0) / 220.0
    electrical_drive = max(0.0, contact_power - 8.0) / 45.0
    eta_contact = getattr(sim.aging_profile, 'eta_effective', 0.0)
    eta_oxide = getattr(sim.aging_profile, 'eta_total', eta_contact)
    humidity_drive = max(0.0, sim.relative_humidity - 0.35) / 0.65

    contact_drive = (1.0 + temp_softening + 1.6 * electrical_drive)
    oxide_drive = (1.0 + 0.8 * temp_softening + 0.5 * electrical_drive + 0.6 * humidity_drive)
    pressure_drop = (
        severity
        * profile.loosen_scale
        * sim.vibration_scale
        * sim.base_loosen_rate
        * (1.0 + 0.65 * eta_contact)
        * contact_drive
        * sim.dt_sim
    )
    oxide_growth = (
        severity
        * profile.oxide_scale
        * sim.oxide_growth_rate
        * (1.0 + 0.55 * eta_oxide)
        * oxide_drive
        * sim.dt_sim
    )

    sim.contact_pressure = max(0.12, sim.contact_pressure - pressure_drop)
    sim.oxide_index = min(4.0, sim.oxide_index + oxide_growth)

    sim.R_loose = sim.R_metal * (1.0 / max(sim.contact_pressure, 0.12) ** 2.35 - 1.0)
    sim.R_oxide = severity * (0.001 + 0.022 * sim.oxide_index)
    sim.R_carbon = severity * 0.018 * sim.carb_track[sim.f_node] ** 2
    runaway = 1.0 + 0.08 * electrical_drive
    contact_resistance = sim.R_metal + sim.R_loose + sim.R_oxide + sim.R_carbon
    sim.states[sim.f_node][2] = min(contact_resistance * runaway, sim.R_contact_ceiling)
