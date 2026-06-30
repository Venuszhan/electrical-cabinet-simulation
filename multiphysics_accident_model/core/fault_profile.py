from dataclasses import dataclass

import pandas as pd

from .cabinet_geometry import CONFIG_DIR


FAULT_SITE_CONFIG = {
    'KM1_DC': {
        'representative_node': 'KM1_DC',
        'circuit_id': 'DC_LOOP',
        'supply_mode': 'DC',
        'component_type': 'contactor_dc_terminal',
    },
    'KM1_AC': {
        'representative_node': 'KM1_AC',
        'circuit_id': 'AC_LOOP',
        'supply_mode': 'AC',
        'component_type': 'contactor_ac_terminal',
    },
    # X1/X2 are terminal-block fault sites represented by the concrete
    # wiring points available in component_table.csv.
    'X1': {
        'representative_node': 'X1_1',
        'circuit_id': 'DC_LOOP',
        'supply_mode': 'DC',
        'component_type': 'terminal_block',
    },
    'X2': {
        'representative_node': 'X2_1',
        'circuit_id': 'AC_LOOP',
        'supply_mode': 'AC',
        'component_type': 'terminal_block',
    },
}

FAULT_NODE_CIRCUITS = {
    site: (config['circuit_id'], config['supply_mode'])
    for site, config in FAULT_SITE_CONFIG.items()
}

DEFAULT_FAULT_SEVERITY = 0.75


@dataclass
class FaultProfile:
    fault_node: str
    fault_type: str
    severity: float
    start_time: float
    loosen_scale: float
    oxide_scale: float
    vibration_scale: float


def get_fault_circuit(fault_node):
    if fault_node not in FAULT_NODE_CIRCUITS:
        known_nodes = ', '.join(sorted(FAULT_NODE_CIRCUITS))
        raise ValueError(f'Unknown fault_node: {fault_node}. Known fault nodes: {known_nodes}')
    return FAULT_NODE_CIRCUITS[fault_node]


def get_fault_site_config(fault_site):
    if fault_site not in FAULT_SITE_CONFIG:
        known_sites = ', '.join(sorted(FAULT_SITE_CONFIG))
        raise ValueError(f'Unknown fault_site: {fault_site}. Known fault sites: {known_sites}')
    return FAULT_SITE_CONFIG[fault_site]


def build_fault_profile(config, fault_node):
    severity = float(config.get('severity', DEFAULT_FAULT_SEVERITY))
    severity = max(0.0, min(1.0, severity))
    return FaultProfile(
        fault_node=fault_node,
        fault_type=config.get('fault_type', 'poor_contact'),
        severity=severity,
        start_time=float(config['t_fault_start']),
        loosen_scale=float(config['contact_loosen_scale']),
        oxide_scale=float(config['oxide_scale']),
        vibration_scale=float(config.get('vibration_scale', 1.0)),
    )


def build_terminal_configs():
    table = pd.read_csv(CONFIG_DIR / 'case_table.csv')
    if 'fault_terminal' not in table.columns and 'fault_node' in table.columns:
        table = table.rename(columns={'fault_node': 'fault_terminal'})
    if 'severity' not in table.columns and 'fault_severity' in table.columns:
        table = table.rename(columns={'fault_severity': 'severity'})
    if 't_fault_start' not in table.columns:
        table['t_fault_start'] = 120.0
    ac_defaults = {
        'ac_source_phase': 0.0,
        'ac_extinction_voltage': 7.0,
        'ac_restrike_voltage': 38.0,
        'ac_reignition_overshoot': 6.0,
        'ac_phase_extinction_band': 0.12,
    }
    for column, default in ac_defaults.items():
        if column not in table.columns:
            table[column] = default
    configs = {}
    for row in table.itertuples(index=False):
        fault_site = getattr(row, 'fault_site', row.fault_terminal)
        site_config = get_fault_site_config(fault_site)
        circuit_id, supply_mode = get_fault_circuit(fault_site)
        configs[row.case_id] = {
            'case_id': row.case_id,
            'current_profile': row.current_profile,
            'ambient_temp': float(row.ambient_temp),
            'vent_state': row.vent_state,
            'door_state': row.door_state,
            't_fault_start': float(row.t_fault_start),
            'fault_terminal': fault_site,
            'fault_site': fault_site,
            'fault_component': fault_site,
            'representative_fault_node': site_config['representative_node'],
            'component_type': site_config['component_type'],
            'fault_type': getattr(row, 'fault_type', 'poor_contact'),
            'severity': float(getattr(row, 'severity', DEFAULT_FAULT_SEVERITY)),
            'vibration_scale': float(getattr(row, 'vibration_scale', 1.0)),
            'circuit_id': circuit_id,
            'supply_mode': supply_mode,
            'source_voltage_nominal': float(row.source_voltage_nominal),
            'upstream_path_resistance': float(row.upstream_path_resistance),
            'fault_pos': (float(row.fault_x_mm), float(row.fault_y_mm), float(row.fault_z_mm)),
            'surface_area': float(row.surface_area),
            'conv_hA': float(row.conv_hA),
            'heat_capacity': float(row.heat_capacity),
            'fuel_mass': float(row.fuel_mass),
            'burn_rate': float(row.burn_rate),
            'contact_ceiling': float(row.contact_ceiling),
            'contact_loosen_scale': float(row.contact_loosen_scale),
            'oxide_scale': float(row.oxide_scale),
            'arc_voltage_base': float(row.arc_voltage_base),
            'arc_field_v_per_mm': float(row.arc_field_v_per_mm),
            'k_arc_ayrton': float(row.k_arc_ayrton),
            'arc_growth_coeff': float(row.arc_growth_coeff),
            'arc_rebridge_coeff': float(row.arc_rebridge_coeff),
            'arc_energy_zone_low': float(row.arc_energy_zone_low),
            'arc_energy_zone_high': float(row.arc_energy_zone_high),
            'gas_ignition_threshold': float(row.gas_ignition_threshold),
            'min_ignition_duration': float(row.min_ignition_duration),
            'min_arc_hold_current': float(row.min_arc_hold_current),
            'spread_plume_scale': float(row.spread_plume_scale),
            'spread_harness_scale': float(row.spread_harness_scale),
            'spread_exposure_gain': float(row.spread_exposure_gain),
            'spread_gas_threshold_scale': float(row.spread_gas_threshold_scale),
            'spread_temp_threshold_scale': float(row.spread_temp_threshold_scale),
            'spread_heat_threshold_scale': float(row.spread_heat_threshold_scale),
            'spread_exposure_threshold_scale': float(row.spread_exposure_threshold_scale),
            'min_spread_signals': int(row.min_spread_signals),
            'spread_current_floor': float(row.spread_current_floor),
            'spread_current_ref_a': float(row.spread_current_ref_a),
            'spread_current_span_a': float(row.spread_current_span_a),
            'current_level': float(getattr(row, 'current_level', 10.0)),
            'service_years': float(getattr(row, 'service_years', 30.0 / 365.0)),
            'random_seed': int(getattr(row, 'random_seed', 0)),
            'link_conductance': {},
        }
        if row.supply_mode == 'AC':
            configs[row.case_id].update({
                'ac_source_phase': float(row.ac_source_phase),
                'ac_extinction_voltage': float(row.ac_extinction_voltage),
                'ac_restrike_voltage': float(row.ac_restrike_voltage),
                'ac_reignition_overshoot': float(row.ac_reignition_overshoot),
                'ac_phase_extinction_band': float(row.ac_phase_extinction_band),
                'spread_link_factor': {
                    ('KM1_DC', 'KM1_AC'): 1.00,
                    ('KM1_AC', 'FR1'): 1.02,
                    ('FR1', 'X1'): 1.05,
                },
            })
    for config in configs.values():
        if config['fault_terminal'] == 'KM1_DC':
            config['link_conductance'] = {
                ('QF1', 'KM1_DC'): 0.020,
                ('KM1_DC', 'KM1_AC'): 0.024,
            }
        elif config['fault_terminal'] == 'KM1_AC':
            config['link_conductance'] = {
                ('KM1_DC', 'KM1_AC'): 0.016,
                ('KM1_AC', 'FR1'): 0.0125,
            }
    return configs


def apply_terminal_configuration(sim):
    config = sim.terminal_configs[sim.case_id]
    sim.fault_site = sim.requested_fault_node or config['fault_site']
    site_config = get_fault_site_config(sim.fault_site)
    sim.fault_component = sim.fault_site
    sim.representative_fault_node = site_config['representative_node']
    sim.component_type = site_config['component_type']
    sim.f_node = sim.representative_fault_node
    if sim.f_node not in sim.nodes:
        raise ValueError(f'fault_node {sim.f_node} is not present in component_table.csv')
    sim.circuit_id, circuit_supply_mode = get_fault_circuit(sim.fault_site)
    sim.fault_profile = build_fault_profile(config, sim.fault_site)
    sim.aging_profile = sim.aging_profiles[sim.f_node]
    sim.gamma = sim.aging_profile.eta_total
    sim.Ea_ref = sim.aging_profile.Ea / 1000.0
    sim.A_pyro = sim.A_pyro_base * sim.aging_profile.pyrolysis_rate_multiplier
    sim.base_loosen_rate = 0.0010 * (1.0 + 0.014 * sim.cur_I0) * sim.aging_profile.contact_pressure_rate_multiplier
    sim.oxide_growth_rate = 0.00035 * sim.aging_profile.oxide_growth_multiplier
    sim.current_profile_name = config['current_profile']
    sim.ambient_temp = config['ambient_temp']
    sim.vent_state = config['vent_state']
    sim.door_state = config['door_state']
    sim.t_fault_start = sim.fault_profile.start_time
    sim.supply_mode = circuit_supply_mode
    sim.arc_model.supply_mode = sim.supply_mode
    sim.source_voltage_nominal = config['source_voltage_nominal']
    sim.upstream_path_resistance = config['upstream_path_resistance']
    if sim.f_node == config['representative_fault_node']:
        sim.node_pos[sim.f_node] = config['fault_pos']
        sim.surface_area[sim.f_node] = config['surface_area']
        sim.conv_hA[sim.f_node] = config['conv_hA']
        sim.C_th[sim.f_node] = config['heat_capacity']
        sim.fuel_mass_total[sim.f_node] = config['fuel_mass']
        sim.remaining_fuel[sim.f_node] = config['fuel_mass']
        sim.burn_base_rate[sim.f_node] = config['burn_rate']
    sim.R_contact_ceiling = config['contact_ceiling']
    sim.contact_loosen_scale = sim.fault_profile.loosen_scale
    sim.oxide_scale = sim.fault_profile.oxide_scale
    sim.vibration_scale = sim.fault_profile.vibration_scale
    sim.arc_voltage_base = config['arc_voltage_base']
    sim.arc_field_v_per_mm = config['arc_field_v_per_mm']
    sim.k_arc_ayrton = config['k_arc_ayrton']
    sim.arc_growth_coeff = config['arc_growth_coeff']
    sim.arc_rebridge_coeff = config['arc_rebridge_coeff']
    sim.arc_energy_zone_low = config['arc_energy_zone_low']
    sim.arc_energy_zone_high = config['arc_energy_zone_high']
    sim.gas_ignition_threshold = config['gas_ignition_threshold']
    sim.min_ignition_duration = config['min_ignition_duration']
    sim.min_arc_hold_current = config['min_arc_hold_current']
    sim.spread_plume_scale = config.get('spread_plume_scale', 1.0)
    sim.spread_harness_scale = config.get('spread_harness_scale', 1.0)
    sim.spread_exposure_gain = config.get('spread_exposure_gain', 1.0)
    sim.spread_gas_threshold_scale = config.get('spread_gas_threshold_scale', 1.0)
    sim.spread_temp_threshold_scale = config.get('spread_temp_threshold_scale', 1.0)
    sim.spread_heat_threshold_scale = config.get('spread_heat_threshold_scale', 1.0)
    sim.spread_exposure_threshold_scale = config.get('spread_exposure_threshold_scale', 1.0)
    sim.min_spread_signals = config.get('min_spread_signals', 1)
    sim.spread_current_floor = config.get('spread_current_floor', 1.0)
    sim.spread_current_ref_a = config.get('spread_current_ref_a', 0.0)
    sim.spread_current_span_a = config.get('spread_current_span_a', 1.0)

    for link, conductance in config['link_conductance'].items():
        sim.link_conductance[link] = conductance

    for link, factor in config.get('spread_link_factor', {}).items():
        sim.spread_link_factor[link] = factor

    if sim.supply_mode == 'AC':
        sim.ac_source_phase = config['ac_source_phase']
        sim.ac_extinction_voltage = config['ac_extinction_voltage']
        sim.ac_restrike_voltage = config['ac_restrike_voltage']
        sim.ac_reignition_overshoot = config['ac_reignition_overshoot']
        sim.ac_phase_extinction_band = config['ac_phase_extinction_band']
    else:
        sim.ac_source_phase = 0.0
