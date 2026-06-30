"""Case 生成器：按参数组合批量生成仿真工况配置。

生成 case_table 风格的配置行，支持：
- 故障节点、电流等级、老化年限、通风状态、随机种子
- 固定控制条件：常值电流、26 degC 环境温度、柜门关闭
"""

import itertools

import pandas as pd

from .cabinet_geometry import CONFIG_DIR
from .fault_profile import FAULT_NODE_CIRCUITS, FAULT_SITE_CONFIG


FAULTABLE_NODES = list(FAULT_SITE_CONFIG.keys())

CURRENT_LEVELS = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

SERVICE_YEARS_LEVELS = [0.0, 5.0, 10.0, 20.0, 30.0]

FIXED_AMBIENT_TEMP = 26.0

VENT_STATES = ['normal', 'reduced', 'forced', 'off']

DOOR_STATES = ['closed']

N_SEEDS_PER_CONFIG = 3

# 从现有 case_table 读取基准参数
_BASE_CASES = None
_COMPONENTS = None


def _load_components():
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    df = pd.read_csv(CONFIG_DIR / 'component_table.csv')
    _COMPONENTS = {row.node: row for row in df.itertuples(index=False)}
    return _COMPONENTS


def _load_base_cases():
    global _BASE_CASES
    if _BASE_CASES is not None:
        return _BASE_CASES
    df = pd.read_csv(CONFIG_DIR / 'case_table.csv')
    _BASE_CASES = {}
    for row in df.itertuples(index=False):
        _BASE_CASES[row.case_id] = {
            'case_id': row.case_id,
            'current_profile': row.current_profile,
            't_fault_start': float(row.t_fault_start),
            'supply_mode': row.supply_mode,
            'source_voltage_nominal': float(row.source_voltage_nominal),
            'upstream_path_resistance': float(row.upstream_path_resistance),
            'fault_x_mm': float(row.fault_x_mm),
            'fault_y_mm': float(row.fault_y_mm),
            'fault_z_mm': float(row.fault_z_mm),
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
        }
        if row.supply_mode == 'AC':
            _BASE_CASES[row.case_id].update({
                'ac_source_phase': float(row.ac_source_phase),
                'ac_extinction_voltage': float(row.ac_extinction_voltage),
                'ac_restrike_voltage': float(row.ac_restrike_voltage),
                'ac_reignition_overshoot': float(row.ac_reignition_overshoot),
                'ac_phase_extinction_band': float(row.ac_phase_extinction_band),
            })
    return _BASE_CASES


def generate_case_table(
    fault_nodes=None,
    current_levels=None,
    service_years_levels=None,
    vent_states=None,
    door_states=None,
    n_seeds=None,
    save_path=None,
):
    """生成 case 配置表。

    参数：
        fault_nodes: 故障节点列表，默认 FAULTABLE_NODES
        current_levels: 电流等级列表，默认 CURRENT_LEVELS
        service_years_levels: 老化年限列表，默认 SERVICE_YEARS_LEVELS
        vent_states: 通风状态列表，默认 VENT_STATES
        door_states: 固定控制条件，默认仅 closed
        n_seeds: 每组配置的随机种子数，默认 N_SEEDS_PER_CONFIG
        save_path: 保存路径，默认不保存

    返回 DataFrame。
    """
    fault_nodes = fault_nodes or FAULTABLE_NODES
    current_levels = current_levels or CURRENT_LEVELS
    service_years_levels = service_years_levels or SERVICE_YEARS_LEVELS
    vent_states = vent_states or VENT_STATES
    door_states = door_states or DOOR_STATES
    n_seeds = n_seeds or N_SEEDS_PER_CONFIG

    base_cases = _load_base_cases()

    rows = []
    case_counter = 0

    for fault_node in fault_nodes:
        circuit_id, supply_mode = FAULT_NODE_CIRCUITS[fault_node]
        site_config = FAULT_SITE_CONFIG[fault_node]
        representative_node = site_config['representative_node']
        component = _load_components()[representative_node]
        base_id = 'DC_CONSTANT' if supply_mode == 'DC' else 'AC_CONSTANT'
        base = base_cases.get(base_id, {})

        for current, service_years, vent, door in itertools.product(
            current_levels, service_years_levels, vent_states, door_states
        ):
            for seed_idx in range(n_seeds):
                case_counter += 1
                ambient_temp = FIXED_AMBIENT_TEMP
                current_profile = 'constant'

                seed = seed_idx + case_counter * 1000

                row = {
                    'case_id': f'CASE_{case_counter:05d}',
                    'base_case_id': base_id,
                    'current_level': current,
                    'current_profile': current_profile,
                    'fault_terminal': fault_node,
                    'fault_site': fault_node,
                    'fault_component': fault_node,
                    'representative_fault_node': representative_node,
                    'component_type': site_config['component_type'],
                    'fault_type': 'poor_contact',
                    'severity': 0.75,
                    'service_years': service_years,
                    'ambient_temp': ambient_temp,
                    'vent_state': vent,
                    'door_state': door,
                    't_fault_start': base.get('t_fault_start', 120.0),
                    'circuit_id': circuit_id,
                    'supply_mode': supply_mode,
                    'source_voltage_nominal': base.get('source_voltage_nominal', 110.0),
                    'upstream_path_resistance': base.get('upstream_path_resistance', 0.012),
                    'fault_x_mm': float(component.x_mm),
                    'fault_y_mm': float(component.y_mm),
                    'fault_z_mm': float(component.z_mm),
                    'surface_area': float(component.surface_area),
                    'conv_hA': float(component.conv_hA),
                    'heat_capacity': float(component.C_th),
                    'fuel_mass': float(component.fuel_mass),
                    'burn_rate': float(component.burn_rate),
                    'contact_ceiling': base.get('contact_ceiling'),
                    'contact_loosen_scale': base.get('contact_loosen_scale'),
                    'oxide_scale': base.get('oxide_scale'),
                    'arc_voltage_base': base.get('arc_voltage_base'),
                    'arc_field_v_per_mm': base.get('arc_field_v_per_mm'),
                    'k_arc_ayrton': base.get('k_arc_ayrton'),
                    'arc_growth_coeff': base.get('arc_growth_coeff'),
                    'arc_rebridge_coeff': base.get('arc_rebridge_coeff'),
                    'arc_energy_zone_low': base.get('arc_energy_zone_low'),
                    'arc_energy_zone_high': base.get('arc_energy_zone_high'),
                    'gas_ignition_threshold': base.get('gas_ignition_threshold'),
                    'min_ignition_duration': base.get('min_ignition_duration'),
                    'min_arc_hold_current': base.get('min_arc_hold_current'),
                    'spread_plume_scale': base.get('spread_plume_scale'),
                    'spread_harness_scale': base.get('spread_harness_scale'),
                    'spread_exposure_gain': base.get('spread_exposure_gain'),
                    'spread_gas_threshold_scale': base.get('spread_gas_threshold_scale'),
                    'spread_temp_threshold_scale': base.get('spread_temp_threshold_scale'),
                    'spread_heat_threshold_scale': base.get('spread_heat_threshold_scale'),
                    'spread_exposure_threshold_scale': base.get('spread_exposure_threshold_scale'),
                    'min_spread_signals': base.get('min_spread_signals'),
                    'spread_current_floor': base.get('spread_current_floor'),
                    'spread_current_ref_a': base.get('spread_current_ref_a'),
                    'spread_current_span_a': base.get('spread_current_span_a'),
                    'ac_source_phase': base.get('ac_source_phase', 0.0),
                    'ac_extinction_voltage': base.get('ac_extinction_voltage', 7.0),
                    'ac_restrike_voltage': base.get('ac_restrike_voltage', 38.0),
                    'ac_reignition_overshoot': base.get('ac_reignition_overshoot', 6.0),
                    'ac_phase_extinction_band': base.get('ac_phase_extinction_band', 0.12),
                    'random_seed': seed,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    df = generate_case_table(save_path=str(CONFIG_DIR / 'generated_cases.csv'))
    print(f'Generated {len(df)} cases.')
    print(df.head())
