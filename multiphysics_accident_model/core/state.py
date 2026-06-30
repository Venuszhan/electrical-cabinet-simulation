import sys

import numpy as np
import pandas as pd

from .aging import aged_initial_resistance, apply_fault_aging_effects, init_aging_parameters
from .cabinet_geometry import (
    AMBIENT_TEMP_C,
    CABINET_DIMS,
    CABINET_ZONES,
    VENT_AREA_M2,
    find_zones_for_point,
    get_dist as geometry_get_dist,
)
from .fault_profile import (
    apply_terminal_configuration,
    build_terminal_configs,
    get_fault_circuit,
    get_fault_site_config,
)
from .graph_builder import build_graph
from .operating_profile import CurrentProfile, apply_operating_case, equivalent_load_resistance
from ..observation.sensor_model import (
    apply_gas_sensor_noise,
    apply_sensor_drift,
    apply_sensor_dropout,
    generate_ar_noise,
)
from ..physics.arc_model import (
    ArcModel,
    compute_ac_arc_branch,
    compute_arc_branch,
    compute_dc_arc_branch,
    get_source_wave,
    required_ignition_energy,
)
from ..physics.contact_model import compute_contact_branch, estimate_contact_power, update_fault_contact
from ..physics.char_model import compute_char_rate
from ..physics.gas_smoke_model import update_gas_smoke_step
from ..physics.ignition_spread import (
    build_diagnostics,
    compute_ignition_probability,
    compute_incident_fire_heat,
    get_burnable_mass,
    get_ignitable_mass,
    get_min_burnable_mass,
    secondary_spread_current_factor,
    spread_coupling,
    update_combustion,
    update_secondary_spread,
)
from ..physics.pyrolysis_model import compute_gas_source_rate
from ..physics.thermal_network import compute_thermal_step

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class PIGAT_Realistic_DigitalTwin_V3:
    """
    电气柜接触器端子接触不良 -> 微弧碳化 -> 稳定电弧 -> 引燃 -> 蔓延 的半物理模型。

    本文件只保留状态初始化和时间推进编排；物理公式拆到 physics/，配置拆到 config/。
    """

    generate_ar_noise = generate_ar_noise
    _apply_terminal_configuration = apply_terminal_configuration
    _get_source_wave = get_source_wave
    _estimate_contact_power = estimate_contact_power
    _compute_contact_branch = compute_contact_branch
    _update_fault_contact = update_fault_contact
    _compute_dc_arc_branch = compute_dc_arc_branch
    _compute_ac_arc_branch = compute_ac_arc_branch
    _compute_arc_branch = compute_arc_branch
    _required_ignition_energy = required_ignition_energy
    _spread_coupling = spread_coupling
    _secondary_spread_current_factor = secondary_spread_current_factor
    _compute_incident_fire_heat = compute_incident_fire_heat
    _update_combustion = update_combustion
    _build_diagnostics = build_diagnostics

    def __init__(self, init_current, fault_terminal='KM1_DC', case_id=None, service_days=30, service_years=None, seed=None, vent_state=None, case_config=None):
        self.rng = np.random.default_rng(seed)

        self.dt_record = 0.02
        self.dt_sim = 0.002
        self.cur_I0 = float(init_current)
        self.U_dc = 110.0
        self.U_ac_rms = 110.0
        self.grid_freq_hz = 50.0
        self.terminal_configs = build_terminal_configs()
        if case_config is not None:
            runtime = dict(case_config)
            runtime_fault_site = str(runtime.get('fault_site', runtime['fault_terminal']))
            _, runtime_supply_mode = get_fault_circuit(runtime_fault_site)
            template_id = 'DC_CONSTANT' if runtime_supply_mode == 'DC' else 'AC_CONSTANT'
            configured = dict(self.terminal_configs[template_id])
            configured.update(runtime)
            runtime_site = get_fault_site_config(runtime_fault_site)
            configured['case_id'] = str(runtime['case_id'])
            configured['fault_terminal'] = runtime_fault_site
            configured['fault_site'] = runtime_fault_site
            configured['fault_component'] = runtime_fault_site
            configured['representative_fault_node'] = runtime_site['representative_node']
            configured['component_type'] = runtime_site['component_type']
            configured['circuit_id'] = runtime_site['circuit_id']
            configured['supply_mode'] = runtime_site['supply_mode']
            configured['fault_pos'] = (
                float(runtime['fault_x_mm']),
                float(runtime['fault_y_mm']),
                float(runtime['fault_z_mm']),
            )
            configured['link_conductance'] = {}
            if runtime_fault_site == 'KM1_DC':
                configured['link_conductance'] = {
                    ('QF1', 'KM1_DC'): 0.020,
                    ('KM1_DC', 'KM1_AC'): 0.024,
                }
            elif runtime_fault_site == 'KM1_AC':
                configured['link_conductance'] = {
                    ('KM1_DC', 'KM1_AC'): 0.016,
                    ('KM1_AC', 'FR1'): 0.0125,
                }
            if runtime_supply_mode == 'AC':
                configured['spread_link_factor'] = {
                    ('KM1_DC', 'KM1_AC'): 1.00,
                    ('KM1_AC', 'FR1'): 1.02,
                    ('FR1', 'X1'): 1.05,
                }
            self.terminal_configs[configured['case_id']] = configured
            case_id = configured['case_id']
        elif vent_state is not None:
            for cfg in self.terminal_configs.values():
                cfg['vent_state'] = vent_state
        self.case_id = case_id
        self.requested_fault_node = fault_terminal if self.case_id is None else None
        if self.case_id is None:
            _, fault_supply_mode = get_fault_circuit(fault_terminal)
            self.case_id = 'DC_CONSTANT' if fault_supply_mode == 'DC' else 'AC_CONSTANT'
        if self.case_id not in self.terminal_configs:
            raise ValueError(f'Unknown case_id: {self.case_id}')
        initial_fault_site = self.requested_fault_node or self.terminal_configs[self.case_id]['fault_site']
        initial_site_config = get_fault_site_config(initial_fault_site)
        self.fault_site = initial_fault_site
        self.fault_component = initial_fault_site
        self.representative_fault_node = initial_site_config['representative_node']
        self.component_type = initial_site_config['component_type']
        self.f_node = self.representative_fault_node
        self.circuit_id = self.terminal_configs[self.case_id]['circuit_id']
        self.supply_mode = self.terminal_configs[self.case_id]['supply_mode']
        self.arc_model = ArcModel(self.supply_mode)
        self.fault_profile = None

        self.cabinet_dims = CABINET_DIMS.copy()
        self.cabinet_zones = CABINET_ZONES.copy()
        self.vent_area_m2 = VENT_AREA_M2
        self.ambient_temp = AMBIENT_TEMP_C
        self.current_profile_name = 'constant'
        self.vent_state = 'normal'
        self.door_state = 'closed'
        self.t_fault_start = 5.0

        graph = build_graph()
        self.node_pos = graph['node_pos']
        self.nodes = graph['nodes']
        self.node_type = graph['node_type']
        self.node_fault_source_level = graph['fault_source_level']
        self.copper_links = graph['copper_links']
        self.link_conductance = graph['link_conductance']
        self.graph_edges = graph['graph_edges']
        self.electrical_edges = graph['electrical_edges']
        self.thermal_edges = graph['thermal_edges']
        self.air_edges = graph['air_edges']
        self.air_neighbors = graph['air_neighbors']
        self.surface_area = graph['surface_area']
        self.conv_hA = graph['conv_hA']
        self.C_th = graph['C_th']
        self.node_resistance = graph['node_resistance']
        self.series_path_nodes = graph['series_path_nodes']
        self.solid_thermal_neighbors = graph['solid_thermal_neighbors']
        self.solid_air_zone = graph['solid_air_zone']
        self.air_zone_members = graph['air_zone_members']
        self.air_zone_air_neighbors = graph['air_zone_air_neighbors']
        self.air_zone_outlet_a = graph['air_zone_outlet_a']
        self.air_zone_nodes = graph['air_zone_nodes']
        self.cabinet_boundary_nodes = graph['cabinet_boundary_nodes']
        self.air_zone_wall_neighbors = graph['air_zone_wall_neighbors']
        self.vent_flow_multiplier = 1.0

        self.R_metal_base = 0.002
        self.R_metal = self.R_metal_base
        init_aging_parameters(self, service_days=service_days, service_years=service_years, rng=self.rng)
        self.states = {n: np.array([self.ambient_temp, 0.0, aged_initial_resistance(self, n)], dtype=float) for n in self.nodes}

        self.sensor_T = {n: self.ambient_temp for n in self.nodes}
        self.tau_sensor = 8.0
        self.k_leak = 6.0
        self.gas_diffusion = 0.12

        self.noise_states = {
            'I': 0.0,
            'V': {n: 0.0 for n in self.nodes},
            'T': {n: 0.0 for n in self.nodes},
            'C': {n: 0.0 for n in self.nodes},
        }
        self.ar_phi = 0.82
        self.sensor_drift = {'I': 0.0, 'T': {n: 0.0 for n in self.nodes}, 'C': {n: 0.0 for n in self.nodes}}

        self.contact_pressure = 1.0
        self.oxide_index = 0.0
        self.base_loosen_rate = 0.0010 * (1.0 + 0.014 * self.cur_I0) * self.aging_profile.contact_pressure_rate_multiplier
        self.oxide_growth_rate = 0.00035 * self.aging_profile.oxide_growth_multiplier
        self.contact_loosen_scale = 1.0
        self.oxide_scale = 1.0
        self.vibration_scale = 1.0
        self.relative_humidity = 0.55
        self.R_contact_ceiling = 0.42 if self.f_node == 'KM1_DC' else 0.36
        self.R_loose = 0.0
        self.R_oxide = 0.0
        self.R_carbon = 0.0
        self.carb_track = {n: 0.0 for n in self.nodes}

        self.fuel_mass_total = graph['fuel_mass_total']
        self.remaining_fuel = self.fuel_mass_total.copy()

        # 累计已热解质量，只做诊断和阶段触发辅助，不直接等于已燃烧质量
        self.pyrolyzed_mass = {n: 0.0 for n in self.nodes}

        # 已热解挥发分/可燃气体前驱物库存
        # 物理含义：固相绝缘材料热解后，不是立刻消失，而是先进入挥发分池
        self.volatile_pool = {n: 0.0 for n in self.nodes}

        # 已热解形成的可燃炭残余，燃烧阶段可继续贡献热释放
        self.char_fuel_mass = {n: 0.0 for n in self.nodes}

        # 气体暴露积分，用于 Stage 1.5 稳健触发
        self.gas_exposure = {n: 0.0 for n in self.nodes}

        # 碳化程度
        self.char_index = {n: 0.0 for n in self.nodes}

        # 热解质量分配参数
        self.pyro_volatile_yield = 0.62
        self.pyro_char_yield = 0.22
        self.pyro_loss_yield = 0.16

        # volatile_pool → C_raw 的释放参数
        self.volatile_release_base = 0.28
        self.volatile_release_temp_scale = 0.85
        self.volatile_pool_cap_factor = 1.20

        # Stage 1.5 辅助触发阈值
        self.volatile_stage_threshold = 0.18
        self.pyro_mass_stage_threshold = 0.45
        self.gas_stage_threshold = 15.0

        # ---------- Stage 1.5 gas/volatile trigger calibration ----------
        # 低浓度气体长期暴露积分
        self.gas_exposure_base = 0.2
        self.gas_exposure_stage_threshold = 25.0

        # C_raw 软门槛：pyrolyzed_mass 只在存在轻微气体信号时辅助触发
        self.gas_stage_soft_threshold = 0.8

        # 挥发分释放暴露：记录 volatile_pool 实际释放过多少
        self.volatile_release_exposure = {n: 0.0 for n in self.nodes}
        self.volatile_release_exposure_threshold = 0.20

        # 挥发分生成与峰值诊断
        self.volatile_generated_mass = {n: 0.0 for n in self.nodes}
        self.volatile_pool_peak = {n: 0.0 for n in self.nodes}

        # ---------- Stage 1.5 trigger snapshot ----------
        self.stage15_trigger_snapshot = None

        # ---------- Stage 1.0 pre-carbonization ----------
        # Stage 1.0 只能形成预碳化/热解残余，不允许把导电碳桥直接推到成熟
        self.stage1_carb_cap = 0.45
        # 原来 Stage 1.0 的 carb_track 增长系数是 0.010，容易在长时间 Stage 1 中饱和
        self.stage1_carb_gain = 0.003

        # ---------- Stage 1.5 conductive carbon bridge growth ----------
        # Stage 1.5 微弧阶段，导电碳桥在弧功率/热解条件下继续成熟
        self.stage15_carb_gain = 0.010
        self.stage15_arc_carb_gain = 0.000020

        # Stage 2 碳桥成熟阈值
        self.stage2_carb_threshold = 0.65

        # ---------- Stage 1.5 -> Stage 2 calibration ----------
        self.carb_track_at_stage15 = None
        self.arc_energy_at_stage15 = 0.0
        self.stage15_start_time = None
        self.stage15_actual_duration = None
        self.min_stage15_duration = 0.8
        self.stage15_to_stage2_arc_energy = 120.0

        # ---------- Stage 3 / combustion validity ----------
        self.volatile_burn_weight = 0.8
        self.char_burn_weight = 0.50
        self.char_ignition_weight = 0.08
        self.min_burnable_mass_abs = 0.15
        self.min_burnable_mass_frac = 0.02

        self.burn_base_rate = graph['burn_base_rate']
        self.heat_of_combustion = 26e6
        self.combustion_efficiency = 0.72

        self.P_crit_contact = 35.0
        self.P_crit_arc_1 = 50.0
        self.P_crit_arc_2 = 250.0
        self.T_pyro = 240.0
        self.T_ig = 330.0
        self.T_spread = 310.0
        self.gas_stage_threshold = 15.0
        self.gas_ignition_threshold = 35.0
        self.gas_spread_threshold = 25.0
        self.gas_index_cap = 120.0
        self.arc_energy_zone_low = 630.0
        self.arc_energy_zone_high = 1515.0
        self.min_ignition_current = 3.0
        self.min_ignition_duration = 1.5
        self.min_arc_hold_current = 1.0
        self.ignition_drive_threshold = 1.5

        self.arc_voltage_base = 16.0
        self.arc_field_v_per_mm = 9.0
        self.k_arc_ayrton = 12.0
        self.arc_growth_coeff = 0.0032
        self.arc_rebridge_coeff = 0.18
        self.arc_gap_mm = 0.0
        self.arc_on = False
        self.arc_voltage = 0.0
        self.arc_current = 0.0
        self.arc_power = 0.0
        self.arc_duration = 0.0
        self.arc_energy_effective = 0.0
        self.arc_quenched = False
        self.arc_off_time = 0.0

        self.source_voltage_nominal = self.U_dc if self.supply_mode == 'DC' else self.U_ac_rms
        self.upstream_path_resistance = 0.0
        self.ac_source_phase = 0.0
        self.ac_extinction_voltage = 7.0
        self.ac_restrike_voltage = 38.0
        self.ac_reignition_overshoot = 6.0
        self.ac_phase_extinction_band = 0.12
        self.ac_arc_conducting = False
        self.ac_zero_cross_total = 0
        self.ac_reignition_total = 0
        self.prev_ac_source_sign = 1

        self.node_hrr = {n: 0.0 for n in self.nodes}
        self.node_ignited = {n: False for n in self.nodes}
        self.node_ignition_time = {n: None for n in self.nodes}
        self.node_ignition_cause = {n: None for n in self.nodes}
        self.ignition_drive = {n: 0.0 for n in self.nodes}
        self.spread_exposure = {n: 0.0 for n in self.nodes}
        self.thermal_spread_duration = {n: 0.0 for n in self.nodes}
        self.min_thermal_spread_duration = 8.0
        self.thermal_spread_temp_margin_c = 55.0
        self.spread_material_factor = graph['spread_material_factor']
        self.spread_heat_threshold = graph['spread_heat_threshold']
        self.spread_exposure_threshold = graph['spread_exposure_threshold']
        self.spread_plume_scale = 1.0
        self.spread_harness_scale = 1.0
        self.spread_exposure_gain = 1.0
        self.spread_gas_threshold_scale = 1.0
        self.spread_temp_threshold_scale = 1.0
        self.spread_heat_threshold_scale = 1.0
        self.spread_exposure_threshold_scale = 1.0
        self.min_spread_signals = 1
        self.spread_current_floor = 1.0
        self.spread_current_ref_a = 0.0
        self.spread_current_span_a = 1.0
        self.spread_link_factor = graph['spread_link_factor']

        self.stage = 1.0
        self.step_counter = 0
        self.stage_sample_counts = {1.0: 0, 1.5: 0, 2.0: 0, 3.0: 0}

        self.peak_arc_power = 0.0
        self.peak_fire_hrr = 0.0
        self.peak_core_temp = self.ambient_temp
        self.peak_T_air_zone = self.ambient_temp
        self.peak_T_solid_non_air = self.ambient_temp
        self.peak_fault_voltage = 0.0
        self.Q_at_ig = None
        self.fault_remaining_fuel_at_stage3 = None
        self.fault_volatile_pool_at_stage3 = None
        self.fault_char_fuel_at_stage3 = None
        self.fault_burnable_mass_at_stage3 = None
        self.fuel_depletion_time = None
        self.solid_fuel_depletion_time = None
        self.burnable_depletion_time = None

        # 如果弧满足引燃条件但无可燃库存，则标记为"无持续燃烧"
        self.arc_without_sustained_fire = False
        self.no_burnable_fire_attempt_time = None

        # Stage 2 触发瞬间快照
        self.stage2_trigger_snapshot = None

        self.last_diagnostics = {}
        self.post_ignition_observation = 120.0

        self._apply_terminal_configuration()
        self.current_profile = CurrentProfile(self.current_profile_name, self.cur_I0, self.t_fault_start, self.rng)
        for node in self.nodes:
            self.states[node][0] = self.ambient_temp
            self.sensor_T[node] = self.ambient_temp
        self.peak_core_temp = self.ambient_temp
        apply_operating_case(self)
        apply_fault_aging_effects(self)

    def get_dist(self, n1, n2):
        return geometry_get_dist(self, n1, n2)

    def get_zones_for_point(self, point_mm):
        return find_zones_for_point(self.cabinet_zones, point_mm)

    def get_node_zones(self, node):
        return self.get_zones_for_point(self.node_pos[node])

    def get_graph_nodes_df(self):
        rows = []
        for idx, node in enumerate(self.nodes):
            x_mm, y_mm, z_mm = self.node_pos[node]
            rows.append({
                'node_id': idx,
                'node_name': node,
                'node_type': self.node_type[node],
                'fault_source_level': self.node_fault_source_level[node],
                'x_mm': x_mm,
                'y_mm': y_mm,
                'z_mm': z_mm,
                'zones': ';'.join(self.get_node_zones(node)),
                'is_series_path': int(node in self.series_path_nodes),
                'base_resistance_ohm': self.node_resistance[node],
                'surface_area_m2': self.surface_area[node],
                'conv_hA_W_per_K': self.conv_hA[node],
                'C_th_J_per_K': self.C_th[node],
                'fuel_mass_g': self.fuel_mass_total[node],
                'burn_base_rate_g_per_s': self.burn_base_rate[node],
                'spread_material_factor': self.spread_material_factor[node],
                'spread_heat_threshold_W': self.spread_heat_threshold[node],
                'spread_exposure_threshold': self.spread_exposure_threshold[node],
                'aging_eta_total': self.aging_profiles[node].eta_total,
                'aging_eta_effective': self.aging_profiles[node].eta_effective,
                'aging_Ea_kJ_per_mol': self.aging_profiles[node].Ea / 1000.0,
            })
        return pd.DataFrame(rows)

    def get_case_config_df(self):
        return pd.DataFrame([{
            'case_id': self.case_id,
            'fault_terminal': self.fault_site,
            'fault_site': self.fault_site,
            'fault_component': self.fault_component,
            'representative_fault_node': self.f_node,
            'circuit_id': self.circuit_id,
            'supply_mode': self.supply_mode,
            'component_type': self.component_type,
            'fault_type': self.fault_profile.fault_type,
            'fault_severity': self.fault_profile.severity,
            'current_profile': self.current_profile_name,
            'ambient_temp': self.ambient_temp,
            'vent_state': self.vent_state,
            'door_state': self.door_state,
            't_fault_start': self.t_fault_start,
            'init_current': self.cur_I0,
            'service_years': self.service_years,
        }])

    def get_aging_profiles_df(self):
        rows = []
        for node in self.nodes:
            row = {'node_name': node}
            row.update(self.aging_profiles[node].to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def get_graph_edges_df(self):
        node_id = {node: idx for idx, node in enumerate(self.nodes)}
        rows = []
        for edge in self.graph_edges:
            rows.append({
                'src_id': node_id[edge['src']],
                'dst_id': node_id[edge['dst']],
                'src': edge['src'],
                'dst': edge['dst'],
                'edge_type': edge['edge_type'],
                'd_mm': round(edge['distance_mm'], 3),
                'k_th': edge['k_th'],
                'g_elec': edge['g_elec'],
                'a_air': edge['a_air'],
                'R_line': edge.get('R_line'),
                'D_gas': edge.get('D_gas'),
                'flow_direction': edge.get('flow_direction'),
                'view_factor': edge.get('view_factor'),
                'topology_weight': edge['topology_weight'],
            })
        return pd.DataFrame(rows)

    def get_electrical_edges_df(self):
        rows = []
        for edge in self.electrical_edges:
            row = edge.copy()
            row['d_mm'] = round(self.get_dist(edge['src'], edge['dst']), 3)
            row['g_elec'] = 1.0 / max(float(edge['R_line']), 1e-6)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_thermal_edges_df(self):
        return pd.DataFrame(self.thermal_edges)

    def get_air_edges_df(self):
        return pd.DataFrame(self.air_edges)

    def run(self, record_history=True):
        history = []
        t = 0.0
        t_stage1_5, t_stage2, t_stage3 = None, None, None
        base_max_duration = 900.0

        while True:
            inst_cache = {n: {'V': [], 'I': [], 'P': []} for n in self.nodes}
            steps = int(self.dt_record / self.dt_sim)
            wave_current_samples = []
            fault_voltage_wave_samples = []
            source_voltage_wave_samples = []
            arc_power_samples = []
            zero_cross_start = self.ac_zero_cross_total
            reignition_start = self.ac_reignition_total

            for _ in range(steps):
                curr_t = t + _ * self.dt_sim
                T_eff = min(self.states[self.f_node][0], 900.0)
                R_healthy_sum = sum(self.states[n][2] for n in self.series_path_nodes if n != self.f_node)
                R_load, profile_current = equivalent_load_resistance(self, curr_t)
                series_resistance = R_load + R_healthy_sum

                line_current_nominal, contact_power_est = self._estimate_contact_power(series_resistance, self.states[self.f_node][2])

                if curr_t >= self.t_fault_start and self.stage < 3.0:
                    self._update_fault_contact(T_eff, contact_power_est)

                R_fault = self.states[self.f_node][2]
                source_voltage_wave = self._get_source_wave(curr_t)

                if self.stage == 1.0:
                    V_fault, I_line, P_fault, wave_current, fault_voltage_wave, source_voltage_wave = self._compute_contact_branch(series_resistance, R_fault, curr_t)
                    self.arc_on = False
                    self.arc_voltage = 0.0
                    self.arc_current = 0.0
                    self.arc_power = 0.0
                    self.arc_off_time = 0.0

                    pyro_drive = (
                        max(0.0, T_eff - self.T_pyro) / 140.0
                        + max(0.0, contact_power_est - self.P_crit_contact) / 60.0
                    )

                    # Stage 1.0 只形成预碳化前驱，不允许导电碳桥成熟到 1.0
                    stage1_carb_cap = getattr(self, "stage1_carb_cap", 0.45)
                    stage1_carb_gain = getattr(self, "stage1_carb_gain", 0.003)

                    self.carb_track[self.f_node] = min(
                        stage1_carb_cap,
                        self.carb_track[self.f_node] + stage1_carb_gain * pyro_drive * self.dt_sim,
                    )

                    C_raw_fault = self.states[self.f_node][1]

                    volatile_ready = (
                        self.volatile_pool[self.f_node] >= self.volatile_stage_threshold
                    )
                    pyro_mass_ready = (
                        self.pyrolyzed_mass[self.f_node] >= self.pyro_mass_stage_threshold
                    )
                    gas_exposure_ready = (
                        self.gas_exposure[self.f_node] >= self.gas_exposure_stage_threshold
                    )
                    volatile_release_ready = (
                        self.volatile_release_exposure[self.f_node]
                        >= self.volatile_release_exposure_threshold
                    )

                    C_hard_ready = C_raw_fault >= self.gas_stage_threshold
                    C_soft_ready = C_raw_fault >= self.gas_stage_soft_threshold

                    # pyrolyzed_mass 不能单独触发 Stage 1.5
                    # 只能在已存在轻微气体/挥发分/释放暴露信号时作为辅助
                    pyro_aux_ready = pyro_mass_ready and (
                        C_soft_ready
                        or volatile_ready
                        or gas_exposure_ready
                        or volatile_release_ready
                    )

                    gas_ready = (
                        C_hard_ready
                        or volatile_ready
                        or gas_exposure_ready
                        or volatile_release_ready
                        or pyro_aux_ready
                    )

                    if contact_power_est >= self.P_crit_contact and T_eff >= self.T_pyro and gas_ready:
                        sources = []
                        if C_hard_ready:
                            sources.append("C_raw_hard")
                        if C_soft_ready:
                            sources.append("C_raw_soft")
                        if volatile_ready:
                            sources.append("volatile_pool")
                        if volatile_release_ready:
                            sources.append("volatile_release")
                        if gas_exposure_ready:
                            sources.append("gas_exposure")
                        if pyro_mass_ready:
                            sources.append("pyrolyzed_mass")
                        if pyro_aux_ready:
                            sources.append("pyro_aux")

                        self.stage15_trigger_snapshot = {
                            "time_s": float(curr_t),
                            "source": "+".join(sources),

                            "contact_power_est": float(contact_power_est),
                            "P_crit_contact": float(self.P_crit_contact),

                            "T_eff": float(T_eff),
                            "T_pyro": float(self.T_pyro),

                            "C_raw_fault": float(C_raw_fault),
                            "gas_stage_threshold": float(self.gas_stage_threshold),
                            "gas_stage_soft_threshold": float(self.gas_stage_soft_threshold),

                            "volatile_pool_fault_g": float(self.volatile_pool[self.f_node]),
                            "volatile_stage_threshold_g": float(self.volatile_stage_threshold),

                            "volatile_release_exposure": float(self.volatile_release_exposure[self.f_node]),
                            "volatile_release_exposure_threshold": float(self.volatile_release_exposure_threshold),

                            "pyrolyzed_mass_fault_g": float(self.pyrolyzed_mass[self.f_node]),
                            "pyro_mass_stage_threshold_g": float(self.pyro_mass_stage_threshold),

                            "gas_exposure_fault": float(self.gas_exposure[self.f_node]),
                            "gas_exposure_stage_threshold": float(self.gas_exposure_stage_threshold),

                            "Gate15_C_Hard_Ready": int(C_hard_ready),
                            "Gate15_C_Soft_Ready": int(C_soft_ready),
                            "Gate15_Volatile_Ready": int(volatile_ready),
                            "Gate15_VolatileRelease_Ready": int(volatile_release_ready),
                            "Gate15_PyroMass_Ready": int(pyro_mass_ready),
                            "Gate15_GasExposure_Ready": int(gas_exposure_ready),
                            "Gate15_PyroAux_Ready": int(pyro_aux_ready),
                        }

                        # 进入 Stage 1.5 时记录碳桥和弧能基线
                        self.carb_track_at_stage15 = float(self.carb_track[self.f_node])
                        self.arc_energy_at_stage15 = float(self.arc_energy_effective)
                        self.stage15_start_time = curr_t

                        self.stage = 1.5
                        t_stage1_5 = curr_t
                        self.arc_gap_mm = 0.05

                elif self.stage == 1.5:
                    V_fault, I_line, P_fault, duty, wave_current, fault_voltage_wave, source_voltage_wave = self._compute_arc_branch(series_resistance, intermittent=True, curr_t=curr_t)
                    self.arc_power = P_fault
                    self.arc_duration += duty * self.dt_sim
                    self.arc_energy_effective += P_fault * self.dt_sim
                    self.arc_gap_mm = np.clip(
                        self.arc_gap_mm + (0.12 + self.arc_growth_coeff * P_fault - self.arc_rebridge_coeff * self.carb_track[self.f_node]) * self.dt_sim,
                        0.05,
                        2.0,
                    )
                    # Stage 1.5：微弧作用下，预碳化路径继续向导电碳桥成熟
                    stage15_carb_gain = getattr(self, "stage15_carb_gain", 0.010)
                    stage15_arc_carb_gain = getattr(self, "stage15_arc_carb_gain", 0.000020)

                    stage15_pyro_drive = (
                        max(0.0, T_eff - self.T_pyro) / 120.0
                        + max(0.0, P_fault - self.P_crit_arc_1) / 180.0
                        + 0.5 * max(0.0, duty)
                    )

                    arc_carb_drive = stage15_arc_carb_gain * max(0.0, P_fault) * duty

                    self.carb_track[self.f_node] = min(
                        1.0,
                        self.carb_track[self.f_node]
                        + stage15_carb_gain * stage15_pyro_drive * self.dt_sim
                        + arc_carb_drive * self.dt_sim,
                    )

                    stage15_duration = curr_t - t_stage1_5 if t_stage1_5 is not None else 0.0

                    if self.carb_track_at_stage15 is None:
                        self.carb_track_at_stage15 = float(self.carb_track[self.f_node])

                    carb_increment = self.carb_track[self.f_node] - self.carb_track_at_stage15
                    arc_energy_since_stage15 = self.arc_energy_effective - self.arc_energy_at_stage15

                    carb_bridge_ready = self.carb_track[self.f_node] >= self.stage2_carb_threshold
                    arc_energy_ready = arc_energy_since_stage15 >= self.stage15_to_stage2_arc_energy
                    duration_ready = stage15_duration >= self.min_stage15_duration

                    if duration_ready and carb_bridge_ready and arc_energy_ready:
                        self.stage = 2.0
                        t_stage2 = curr_t
                        self.stage15_actual_duration = curr_t - t_stage1_5
                        self.arc_gap_mm = max(self.arc_gap_mm, 0.15)
                        self.stage2_trigger_snapshot = {
                            "time_s": float(curr_t),
                            "carb_track": float(self.carb_track[self.f_node]),
                            "stage2_carb_threshold": float(self.stage2_carb_threshold),
                            "arc_energy_since_stage15": float(arc_energy_since_stage15),
                            "stage15_to_stage2_arc_energy": float(self.stage15_to_stage2_arc_energy),
                            "stage15_duration": float(stage15_duration),
                            "min_stage15_duration": float(self.min_stage15_duration),
                            "Stage2_Carb_Ready_at_Stage2": int(carb_bridge_ready),
                            "Stage2_ArcEnergy_Ready_at_Stage2": int(arc_energy_ready),
                            "Stage2_Duration_Ready_at_Stage2": int(duration_ready),
                        }

                elif self.stage == 2.0:
                    V_fault, I_line, P_fault, duty, wave_current, fault_voltage_wave, source_voltage_wave = self._compute_arc_branch(series_resistance, intermittent=False, curr_t=curr_t)
                    self.arc_power = P_fault
                    self.arc_duration += duty * self.dt_sim
                    self.arc_energy_effective += P_fault * self.dt_sim
                    self.arc_gap_mm = np.clip(
                        self.arc_gap_mm + (0.20 + self.arc_growth_coeff * P_fault - 0.40 * self.arc_rebridge_coeff * self.carb_track[self.f_node]) * self.dt_sim,
                        0.10,
                        6.0,
                    )
                    self.carb_track[self.f_node] = min(
                        1.0,
                        self.carb_track[self.f_node]
                        + (0.015 + max(0.0, P_fault - self.P_crit_arc_1) / 1800.0) * self.dt_sim,
                    )

                    if duty < 0.05 or P_fault < 1.0:
                        self.arc_off_time += self.dt_sim
                    else:
                        self.arc_off_time = 0.0

                    if self.supply_mode == 'DC':
                        if I_line < self.min_arc_hold_current or self.arc_gap_mm >= 5.5:
                            self.arc_quenched = True
                            self.arc_power = 0.0
                            self.arc_on = False
                    elif self.arc_off_time > 0.08 and self.arc_gap_mm >= 3.2 and self.carb_track[self.f_node] < 0.55:
                        self.arc_quenched = True
                        self.arc_power = 0.0
                        self.arc_on = False

                    # 无持续燃烧退出保护：如果已标记无可燃库存超过 5s，强制熄弧
                    if (
                        self.arc_without_sustained_fire
                        and self.no_burnable_fire_attempt_time is not None
                        and curr_t - self.no_burnable_fire_attempt_time > 5.0
                    ):
                        self.arc_quenched = True
                        self.arc_power = 0.0
                        self.arc_on = False

                    if t_stage3 is None and self.arc_duration >= self.min_ignition_duration:
                        p_ign = compute_ignition_probability(self, self.f_node)
                        self.ignition_drive[self.f_node] += p_ign * self.dt_sim
                        ignition_condition = (
                            self.ignition_drive[self.f_node]
                            >= self.ignition_drive_threshold
                        )

                        burnable_fault = get_ignitable_mass(self, self.f_node)
                        min_burnable = get_min_burnable_mass(self, self.f_node)
                        burnable_ready = burnable_fault >= min_burnable

                        if ignition_condition and burnable_ready:
                            self.stage = 3.0
                            t_stage3 = curr_t
                            self.Q_at_ig = self.arc_energy_effective
                            self.node_ignited[self.f_node] = True
                            self.node_ignition_time[self.f_node] = curr_t
                            self.node_ignition_cause[self.f_node] = "primary_arc"
                            self.fault_remaining_fuel_at_stage3 = self.remaining_fuel[self.f_node]
                            self.fault_volatile_pool_at_stage3 = self.volatile_pool[self.f_node]
                            self.fault_char_fuel_at_stage3 = self.char_fuel_mass[self.f_node]
                            self.fault_burnable_mass_at_stage3 = burnable_fault

                        elif ignition_condition and not burnable_ready:
                            # 达到弧能/温度/概率条件，但没有有效可燃库存
                            self.arc_without_sustained_fire = True
                            if self.no_burnable_fire_attempt_time is None:
                                self.no_burnable_fire_attempt_time = curr_t
                            # 继续保持 Stage 2，不标记 Stage 3，不设置 node_ignited

                else:
                    I_line = 0.0
                    V_fault = 0.0
                    P_fault = 0.0
                    self.arc_power = 0.0
                    self.arc_off_time = 0.0
                    self.arc_on = False
                    self.arc_voltage = 0.0
                    self.arc_current = 0.0
                    wave_current = 0.0
                    fault_voltage_wave = 0.0

                self.peak_arc_power = max(self.peak_arc_power, self.arc_power)
                self.peak_fault_voltage = max(self.peak_fault_voltage, abs(fault_voltage_wave))
                wave_current_samples.append(wave_current)
                fault_voltage_wave_samples.append(fault_voltage_wave)
                source_voltage_wave_samples.append(source_voltage_wave)
                arc_power_samples.append(self.arc_power)

                for n in self.nodes:
                    if n == self.f_node:
                        v_n = V_fault
                        i_n = I_line
                    elif n in self.series_path_nodes:
                        v_n = I_line * self.states[n][2]
                        i_n = I_line
                    else:
                        v_n = 0.0
                        i_n = 0.0
                    inst_cache[n]['V'].append(v_n)
                    inst_cache[n]['I'].append(i_n)
                    inst_cache[n]['P'].append(v_n * i_n)

            self._update_combustion()
            next_T, pyro_rates, last_fire_heat = compute_thermal_step(self, inst_cache)
            gas_source_rates = {n: compute_gas_source_rate(self, n, pyro_rates[n]) for n in self.nodes}
            char_rates = {n: compute_char_rate(self, n, pyro_rates[n]) for n in self.nodes}
            update_gas_smoke_step(self, next_T, pyro_rates, gas_source_rates, char_rates)
            update_secondary_spread(self, last_fire_heat, t)

            if self.fuel_depletion_time is None and self.remaining_fuel[self.f_node] <= 0.0:
                self.fuel_depletion_time = t + self.dt_record

            if self.solid_fuel_depletion_time is None and self.remaining_fuel[self.f_node] <= 1e-9:
                self.solid_fuel_depletion_time = t + self.dt_record

            if self.burnable_depletion_time is None:
                if get_ignitable_mass(self, self.f_node) <= get_min_burnable_mass(self, self.f_node):
                    self.burnable_depletion_time = t + self.dt_record

            self.step_counter += 1
            base_i_std = 0.30 if self.stage == 1.5 else (0.16 if self.stage >= 2.0 else 0.03)
            base_v_std = 0.50 if self.stage == 1.5 else (0.24 if self.stage >= 2.0 else 0.015)

            i_noise = self.generate_ar_noise('I', std=base_i_std)
            current_wave_array = np.array(wave_current_samples, dtype=float)
            fault_voltage_wave_array = np.array(fault_voltage_wave_samples, dtype=float)
            source_voltage_wave_array = np.array(source_voltage_wave_samples, dtype=float)
            arc_power_array = np.array(arc_power_samples, dtype=float)
            line_current_true = float(np.sqrt(np.mean(np.square(current_wave_array))))
            fault_voltage_true = float(np.sqrt(np.mean(np.square(fault_voltage_wave_array))))
            source_voltage_true = float(np.sqrt(np.mean(np.square(source_voltage_wave_array))))

            # sensor drift updates
            self.sensor_drift['I'] = apply_sensor_drift(self.sensor_drift['I'], drift_rate=0.0008, rng=self.rng)
            for n in self.nodes:
                self.sensor_drift['T'][n] = apply_sensor_drift(
                    self.sensor_drift['T'][n], drift_rate=0.0003, rng=self.rng
                )
                self.sensor_drift['C'][n] = apply_sensor_drift(
                    self.sensor_drift['C'][n], drift_rate=0.0015, rng=self.rng
                )

            if record_history:
                row = {
                    'Time': round(t + self.dt_record, 3),
                    'Stage': self.stage,
                    'Fault_Terminal': self.fault_site,
                    'Fault_Site': self.fault_site,
                    'Representative_Fault_Node': self.f_node,
                    'Fault_Component': self.fault_component,
                    'Component_Type': self.component_type,
                    'Fault_Type': self.fault_profile.fault_type,
                    'Fault_Severity': round(self.fault_profile.severity, 3),
                    'Circuit_ID': self.circuit_id,
                    'Case_ID': self.case_id,
                    'Current_Profile': self.current_profile_name,
                    'Profile_Current_Setpoint': round(profile_current, 3),
                    'Ambient_Temp': round(self.ambient_temp, 2),
                    'Vent_State': self.vent_state,
                    'Door_State': self.door_state,
                    'T_Fault_Start': round(self.t_fault_start, 3),
                    'Supply_Mode': self.supply_mode,
                    'Line_Current': apply_sensor_dropout(
                        round(max(0.0, line_current_true + i_noise + self.sensor_drift['I']), 3),
                        dropout_prob=0.012, rng=self.rng,
                    ),
                    'Fault_Voltage_RMS': round(fault_voltage_true, 3),
                    'Source_Voltage_RMS': round(source_voltage_true, 3),
                    'Fault_Resistance': round(self.states[self.f_node][2], 5),
                    'R_Loose': round(self.R_loose, 6),
                    'R_Oxide': round(self.R_oxide, 6),
                    'R_Carbon': round(self.R_carbon, 6),
                    'Fault_C_Raw': round(self.states[self.f_node][1], 4),
                    'Fault_Volatile_Pool_g': round(self.volatile_pool[self.f_node], 5),
                    'Fault_Char_Fuel_g': round(self.char_fuel_mass[self.f_node], 5),
                    'Fault_Pyrolyzed_Mass_g': round(self.pyrolyzed_mass[self.f_node], 5),
                    'Fault_Remaining_Fuel_g': round(self.remaining_fuel[self.f_node], 5),
                    'Fault_Gas_Exposure': round(self.gas_exposure[self.f_node], 4),
                    'Gate15_C_Raw_Ready': int(
                        self.states[self.f_node][1] >= self.gas_stage_threshold
                    ),
                    'Gate15_Volatile_Ready': int(
                        self.volatile_pool[self.f_node] >= self.volatile_stage_threshold
                    ),
                    'Gate15_PyroMass_Ready': int(
                        self.pyrolyzed_mass[self.f_node] >= self.pyro_mass_stage_threshold
                    ),
                    'Gate15_GasExposure_Ready': int(
                        self.gas_exposure[self.f_node] >= self.gas_exposure_stage_threshold
                    ),
                    'Fault_Volatile_Pool_Peak_g': round(self.volatile_pool_peak[self.f_node], 5),
                    'Fault_Volatile_Generated_g': round(self.volatile_generated_mass[self.f_node], 5),
                    'Fault_Volatile_Release_Exposure_g': round(self.volatile_release_exposure[self.f_node], 5),
                    'Gate15_C_Hard_Ready': int(
                        self.states[self.f_node][1] >= self.gas_stage_threshold
                    ),
                    'Gate15_C_Soft_Ready': int(
                        self.states[self.f_node][1] >= self.gas_stage_soft_threshold
                    ),
                    'Gate15_VolatileRelease_Ready': int(
                        self.volatile_release_exposure[self.f_node]
                        >= self.volatile_release_exposure_threshold
                    ),
                    'Stage15_Duration_s': (
                        round(curr_t - t_stage1_5, 4)
                        if t_stage1_5 is not None and self.stage == 1.5
                        else (round(self.stage15_actual_duration, 4) if self.stage15_actual_duration is not None else 0.0)
                    ),
                    'Stage15_Carb_Increment': (
                        round(self.carb_track[self.f_node] - self.carb_track_at_stage15, 5)
                        if self.carb_track_at_stage15 is not None
                        else 0.0
                    ),
                    'Stage15_Arc_Energy_Increment': (
                        round(self.arc_energy_effective - self.arc_energy_at_stage15, 3)
                        if self.carb_track_at_stage15 is not None
                        else 0.0
                    ),
                    'Carb_Track_Fault': round(self.carb_track[self.f_node], 5),
                    'Carb_Track_At_Stage15': (
                        round(self.carb_track_at_stage15, 5)
                        if self.carb_track_at_stage15 is not None else None
                    ),
                    'Stage2_Carb_Ready': int(
                        self.carb_track[self.f_node] >= self.stage2_carb_threshold
                    ),
                    'Stage2_ArcEnergy_Ready': int(
                        self.carb_track_at_stage15 is not None
                        and (self.arc_energy_effective - self.arc_energy_at_stage15)
                        >= self.stage15_to_stage2_arc_energy
                    ),
                    'Stage2_Duration_Ready': int(
                        t_stage1_5 is not None
                        and (curr_t - t_stage1_5) >= self.min_stage15_duration
                    ),
                    'Arc_On': int(self.arc_on),
                    'Arc_Voltage': round(self.arc_voltage, 3),
                    'Arc_Current': round(self.arc_current, 3),
                    'Arc_Power': round(float(np.mean(arc_power_array)), 3),
                    'Arc_Power_Peak': round(float(np.max(arc_power_array)), 3),
                    'Arc_Gap_mm': round(self.arc_gap_mm, 4),
                    'Arc_Duty': round(float(np.mean(arc_power_array > 1e-6)), 3),
                    'Arc_Energy': round(self.arc_energy_effective, 3),
                    'Effective_Arc_Energy': round(self.arc_energy_effective, 3),
                    'Zero_Crossings_Total': self.ac_zero_cross_total if self.supply_mode == 'AC' else 0,
                    'Reignitions_Total': self.ac_reignition_total if self.supply_mode == 'AC' else 0,
                    'Gas_Concentration': round(self.states[self.f_node][1], 3),
                    'Ignition_Probability': round(compute_ignition_probability(self, self.f_node), 4),
                    'Fire_HRR_Total': round(sum(self.node_hrr.values()), 3),
                    'Zero_Crossings_Window': self.ac_zero_cross_total - zero_cross_start if self.supply_mode == 'AC' else 0,
                    'Reignitions_Window': self.ac_reignition_total - reignition_start if self.supply_mode == 'AC' else 0,
                }

                for n in self.nodes:
                    v_samples = np.array(inst_cache[n]['V'], dtype=float)
                    if self.supply_mode == 'AC':
                        v_true = float(np.sqrt(np.mean(np.square(v_samples))))
                    else:
                        v_true = float(np.mean(v_samples))
                    v_noise = self.generate_ar_noise('V', sub_key=n, std=base_v_std)
                    row[f'{n}_V'] = apply_sensor_dropout(
                        round(v_true + v_noise, 3), dropout_prob=0.008, rng=self.rng,
                    )

                    t_noise = self.generate_ar_noise('T', sub_key=n, std=0.18)
                    row[f'{n}_T'] = apply_sensor_dropout(
                        round(self.sensor_T[n] + t_noise + self.sensor_drift['T'][n], 2),
                        dropout_prob=0.005, rng=self.rng,
                    )
                    row[f'{n}_T_Core'] = round(self.states[n][0], 2)

                    c_cap = getattr(self, 'gas_index_cap', 120.0)
                    c_raw = self.states[n][1] + self.sensor_drift['C'][n]
                    c_obs = c_cap * (1.0 - np.exp(-max(c_raw, 0.0) / c_cap))
                    c_meas = apply_gas_sensor_noise(
                        c_obs,
                        rng=self.rng, base_std=0.6, low_conc_boost=3.0, saturation_level=c_cap * 1e3,
                    )
                    row[f'{n}_C'] = apply_sensor_dropout(
                        round(c_meas, 3), dropout_prob=0.018, rng=self.rng,
                    )
                    row[f'{n}_Char'] = round(self.char_index[n], 4)
                    row[f'{n}_HRR'] = round(self.node_hrr[n], 3)

                history.append(row)
            self.stage_sample_counts[self.stage] = self.stage_sample_counts.get(self.stage, 0) + 1

            t += self.dt_record
            if t_stage3 is None and t >= base_max_duration:
                break
            if t_stage3 is not None and t > t_stage3 + self.post_ignition_observation:
                break
            if self.arc_quenched and t_stage3 is None and self.stage >= 2.0 and t > (t_stage2 or t) + 20.0:
                break

        history_df = pd.DataFrame(history) if record_history else pd.DataFrame()
        diagnostics = self._build_diagnostics(t_stage1_5, t_stage2, t_stage3, history_df)
        self.last_diagnostics = diagnostics
        return history_df, t_stage1_5, t_stage2, t_stage3, self.Q_at_ig, diagnostics
