import numpy as np


# --- Ignition probability parameters ---
W_IGN_TEMP = 5.0
W_IGN_CONC = 4.0
W_IGN_ENERGY = 3.0
W_IGN_FUEL = 2.0
W_IGN_OXYGEN = 1.5


def spread_coupling(sim, src, node):
    dist_m = sim.get_dist(node, src) / 1000.0
    pair = (src, node) if (src, node) in sim.spread_link_factor else (node, src)
    link_factor = sim.spread_link_factor.get(pair, 0.78)
    vertical_factor = 1.18 if sim.node_pos[src][2] >= sim.node_pos[node][2] else 0.92
    harness_factor = 1.32 if ((src, node) in sim.copper_links or (node, src) in sim.copper_links) else 1.0
    axial_decay = np.exp(-dist_m / 0.34)
    return link_factor * vertical_factor * harness_factor * axial_decay


def secondary_spread_current_factor(sim):
    if sim.supply_mode != 'AC':
        return 1.0
    normalized = np.clip((sim.cur_I0 - sim.spread_current_ref_a) / max(sim.spread_current_span_a, 1e-6), 0.0, 1.0)
    return sim.spread_current_floor + (1.0 - sim.spread_current_floor) * normalized


def compute_incident_fire_heat(sim, node):
    heat_in = 0.18 * sim.node_hrr[node]
    current_factor = sim._secondary_spread_current_factor()
    for src in sim.nodes:
        if src == node or sim.node_hrr[src] <= 0.0:
            continue
        dist_m = sim.get_dist(node, src) / 1000.0
        view_factor = min(0.65, sim.surface_area[src] / max(0.08, 4 * np.pi * dist_m ** 2))
        spread_coupling_value = sim._spread_coupling(src, node)
        radiant_heat = 0.22 * sim.node_hrr[src] * view_factor
        plume_heat = current_factor * sim.spread_plume_scale * 0.10 * sim.node_hrr[src] * spread_coupling_value
        harness_heat = 0.0
        if (src, node) in sim.copper_links or (node, src) in sim.copper_links:
            harness_heat = current_factor * sim.spread_harness_scale * 0.06 * sim.node_hrr[src] * spread_coupling_value
        heat_in += radiant_heat + plume_heat + harness_heat
    return heat_in


def get_burnable_mass(sim, node):
    """有效可燃库存。

    remaining_fuel: 未热解固相燃料
    volatile_pool: 已热解挥发分 / 可燃气体前驱物
    char_fuel_mass: 热解后仍可氧化放热的炭质残余

    volatile_pool 按 0.8 折算，是因为挥发分能量密度、可燃性和固相燃料不完全等价。
    """
    volatile_weight = getattr(sim, "volatile_burn_weight", 0.8)
    char_weight = getattr(sim, "char_burn_weight", 0.50)
    remaining = max(0.0, sim.remaining_fuel.get(node, 0.0))
    volatile = max(0.0, getattr(sim, "volatile_pool", {}).get(node, 0.0))
    char_fuel = max(0.0, getattr(sim, "char_fuel_mass", {}).get(node, 0.0))
    return remaining + volatile_weight * volatile + char_weight * char_fuel


def get_ignitable_mass(sim, node):
    """Fuel capable of initiating flaming ignition.

    Char residue can sustain a developed fire, but it contributes weakly to
    first flaming ignition compared with intact insulation and volatiles.
    """
    volatile_weight = getattr(sim, "volatile_burn_weight", 0.8)
    char_weight = getattr(sim, "char_ignition_weight", 0.08)
    remaining = max(0.0, sim.remaining_fuel.get(node, 0.0))
    volatile = max(0.0, getattr(sim, "volatile_pool", {}).get(node, 0.0))
    char_fuel = max(0.0, getattr(sim, "char_fuel_mass", {}).get(node, 0.0))
    return remaining + volatile_weight * volatile + char_weight * char_fuel


def get_min_burnable_mass(sim, node):
    """进入有效燃烧所需的最小可燃库存。"""
    abs_min = getattr(sim, "min_burnable_mass_abs", 0.15)
    frac_min = getattr(sim, "min_burnable_mass_frac", 0.02)
    return max(abs_min, frac_min * max(sim.fuel_mass_total.get(node, 0.0), 1e-6))


def compute_ignition_probability(sim, node):
    """连续概率引燃：基于温度、气体浓度、弧能和有效可燃库存的 sigmoid 模型。"""
    T_i = sim.states[node][0]
    C_i = sim.states[node][1]
    E_arc = sim.arc_energy_effective

    burnable_i = get_ignitable_mass(sim, node)
    fuel_max = max(sim.fuel_mass_total[node], 1e-6)

    T_ig = sim.T_ig
    C_crit = sim.gas_ignition_threshold
    E_crit = sim.arc_energy_zone_low
    oxygen = np.clip(getattr(sim, "vent_oxygen_scale", 1.0), 0.4, 1.15)

    z = (
        W_IGN_TEMP * (T_i - T_ig) / max(T_ig, 50.0)
        + W_IGN_CONC * (C_i - C_crit) / max(C_crit, 1.0)
        + W_IGN_ENERGY * (E_arc - E_crit) / max(E_crit, 1.0)
        + W_IGN_FUEL * (burnable_i / fuel_max)
        + W_IGN_OXYGEN * (oxygen - 1.0)
    )
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))


def update_combustion(sim):
    oxygen_factor = 0.55 + 0.45 * np.tanh(35.0 * sim.vent_area_m2)
    oxygen_factor *= getattr(sim, "vent_oxygen_scale", 1.0)

    for node in sim.nodes:
        burnable_mass = get_burnable_mass(sim, node)

        if not sim.node_ignited[node] or burnable_mass <= 1e-9:
            sim.node_hrr[node] = 0.0
            continue

        C_i = sim.states[node][1]
        gas_factor = C_i / (C_i + 60.0)
        heat_feedback = np.clip((sim.states[node][0] - sim.T_pyro) / 220.0, 0.0, 2.0)

        burn_rate = (
            sim.burn_base_rate[node]
            * (0.45 + gas_factor + 0.25 * heat_feedback)
            * oxygen_factor
            * sim.aging_profiles[node].combustion_rate_multiplier
        )

        # 不能超过有效可燃库存
        burn_rate = min(burn_rate, burnable_mass / sim.dt_record)

        dm_burn = burn_rate * sim.dt_record

        # 优先消耗 volatile_pool，再消耗 remaining_fuel
        # volatile_pool 是 g，按真实质量扣；burnable_mass 中它按 volatile_burn_weight 折算。
        volatile_weight = getattr(sim, "volatile_burn_weight", 0.8)

        dm_from_vol_equiv = min(volatile_weight * sim.volatile_pool[node], 0.65 * dm_burn)
        dm_from_vol = dm_from_vol_equiv / max(volatile_weight, 1e-6)
        dm_from_vol = min(dm_from_vol, sim.volatile_pool[node])
        sim.volatile_pool[node] -= dm_from_vol

        dm_remaining_equiv = max(0.0, dm_burn - dm_from_vol_equiv)
        dm_from_solid = min(sim.remaining_fuel[node], dm_remaining_equiv)
        sim.remaining_fuel[node] -= dm_from_solid
        dm_remaining_equiv -= dm_from_solid

        char_weight = getattr(sim, "char_burn_weight", 0.50)
        dm_from_char_equiv = min(char_weight * sim.char_fuel_mass[node], dm_remaining_equiv)
        dm_from_char = dm_from_char_equiv / max(char_weight, 1e-6)
        sim.char_fuel_mass[node] -= dm_from_char

        sim.volatile_pool[node] = max(0.0, sim.volatile_pool[node])
        sim.remaining_fuel[node] = max(0.0, sim.remaining_fuel[node])
        sim.char_fuel_mass[node] = max(0.0, sim.char_fuel_mass[node])

        # HRR 按等效燃烧质量计算
        actual_burn_equiv = dm_from_vol_equiv + dm_from_solid + dm_from_char_equiv
        sim.node_hrr[node] = (
            actual_burn_equiv / sim.dt_record
            / 1000.0
            * sim.heat_of_combustion
            * sim.combustion_efficiency
        )

    sim.peak_fire_hrr = max(sim.peak_fire_hrr, sum(sim.node_hrr.values()))


def update_secondary_spread(sim, last_fire_heat, current_t):
    if sim.stage < 3.0:
        return
    current_factor = sim._secondary_spread_current_factor()
    for node in sim.nodes:
        if sim.node_ignited[node] or get_burnable_mass(sim, node) <= get_min_burnable_mass(sim, node):
            continue
        gas_threshold = sim.gas_spread_threshold * sim.spread_gas_threshold_scale * sim.spread_material_factor[node]
        temp_threshold = sim.T_spread * sim.spread_temp_threshold_scale * sim.spread_material_factor[node]
        heat_threshold = sim.spread_heat_threshold_scale * sim.spread_heat_threshold[node]
        spread_drive = current_factor * sim.spread_exposure_gain * max(0.0, last_fire_heat[node] - 0.35 * heat_threshold)
        if sim.states[node][0] >= sim.T_pyro:
            spread_drive += current_factor * sim.spread_exposure_gain * 1.6 * (sim.states[node][0] - sim.T_pyro)
        if sim.states[node][1] >= gas_threshold:
            spread_drive += current_factor * sim.spread_exposure_gain * 60.0 * (sim.states[node][1] - gas_threshold)
        sim.spread_exposure[node] += max(0.0, spread_drive) * sim.dt_record
        temp_ready = sim.states[node][0] >= temp_threshold
        heat_ready = last_fire_heat[node] >= heat_threshold
        exposure_ready = sim.spread_exposure[node] >= sim.spread_exposure_threshold_scale * sim.spread_exposure_threshold[node]
        spread_signal_count = int(temp_ready) + int(heat_ready) + int(exposure_ready)
        material_ignition_temp = max(sim.T_spread, sim.T_ig)
        if sim.states[node][0] >= material_ignition_temp:
            sim.thermal_spread_duration[node] += sim.dt_record
        else:
            sim.thermal_spread_duration[node] = max(
                0.0, sim.thermal_spread_duration[node] - sim.dt_record
            )
        gas_path_ready = (
            sim.states[node][1] >= gas_threshold
            and spread_signal_count >= sim.min_spread_signals
        )
        thermal_path_ready = (
            sim.thermal_spread_duration[node] >= sim.min_thermal_spread_duration
            and (
                heat_ready
                or exposure_ready
                or sim.states[node][0]
                >= material_ignition_temp + sim.thermal_spread_temp_margin_c
            )
        )
        if gas_path_ready or thermal_path_ready:
            sim.node_ignited[node] = True
            sim.node_ignition_time[node] = current_t + sim.dt_record
            sim.node_ignition_cause[node] = (
                "gas_spread" if gas_path_ready else "thermal_surface_spread"
            )


def build_diagnostics(sim, t_stage1_5, t_stage2, t_stage3, history_df=None):
    capability_tags = []
    if sim.stage_sample_counts[1.5] >= 5:
        capability_tags.append('微弧可观测')
    if sim.supply_mode == 'DC' and sim.peak_arc_power > 0.0:
        capability_tags.append('直流连续稳弧')
    if sim.supply_mode == 'AC' and sim.ac_reignition_total > 0:
        capability_tags.append('交流过零重燃')
    if sim.peak_fire_hrr > 0.0 and sim.peak_fire_hrr < 6000.0:
        capability_tags.append('火源非固定')
    if sum(sim.node_ignited.values()) > 1:
        capability_tags.append('支持蔓延')
    if not capability_tags:
        capability_tags.append('仅支持趋势分析')

    limitation_tags = []
    if sim.stage_sample_counts[1.5] < 5:
        limitation_tags.append('微弧采样仍偏短')
    if sim.arc_quenched and t_stage3 is None:
        limitation_tags.append('稳弧持续时间对参数敏感')
    if sim.supply_mode == 'AC' and sim.ac_reignition_total == 0 and t_stage2 is not None:
        limitation_tags.append('交流重燃分辨不足')
    if sim.peak_fire_hrr >= 6000.0:
        limitation_tags.append('热释放仍需试验标定')
    if sum(sim.node_ignited.values()) <= 1:
        limitation_tags.append('蔓延主要受简化热反馈控制')
    if not limitation_tags:
        limitation_tags.append('仍缺少元件级试验标定')

    result = {
        'Fault_Terminal': sim.fault_site,
        'Fault_Site': sim.fault_site,
        'Representative_Fault_Node': sim.f_node,
        'Fault_Component': sim.fault_component,
        'Component_Type': sim.component_type,
        'Fault_Type': sim.fault_profile.fault_type,
        'Fault_Severity': round(sim.fault_profile.severity, 3),
        'Fault_Circuit_ID': sim.circuit_id,
        'Case_ID': sim.case_id,
        'Current_Profile': sim.current_profile_name,
        'Ambient_Temp_C': round(sim.ambient_temp, 2),
        'Vent_State': sim.vent_state,
        'Door_State': sim.door_state,
        'T_Fault_Start_s': round(sim.t_fault_start, 3),
        'Supply_Mode': sim.supply_mode,
        'Service_Years': round(sim.service_years, 3),
        'Aging_Eta_Total': round(sim.aging_profile.eta_total, 4),
        'Aging_Eta_Effective': round(sim.aging_profile.eta_effective, 4),
        'Aging_Ea_kJ_per_mol': round(sim.aging_profile.Ea / 1000.0, 2),
        'Aging_Ignition_Temp_Rise_C': round(sim.aging_profile.ignition_temp_rise_c, 3),
        'Aging_Combustion_Rate_Multiplier': round(sim.aging_profile.combustion_rate_multiplier, 4),
        'Stage_1_5_Time_s': round(t_stage1_5, 3) if t_stage1_5 is not None else None,
        'Stage_2_Time_s': round(t_stage2, 3) if t_stage2 is not None else None,
        'Stage_3_Time_s': round(t_stage3, 3) if t_stage3 is not None else None,
        'Stage_1_5_Samples': int(sim.stage_sample_counts[1.5]),
        'Stage_2_Samples': int(sim.stage_sample_counts[2.0]),
        'Arc_Duration_s': round(sim.arc_duration, 3),
        'Arc_On': bool(sim.arc_on),
        'Arc_Gap_mm': round(sim.arc_gap_mm, 4),
        'Arc_Voltage_V': round(sim.arc_voltage, 3),
        'Arc_Current_A': round(sim.arc_current, 3),
        'Arc_Power_W': round(sim.arc_power, 3),
        'Arc_Energy': round(sim.arc_energy_effective, 1),
        'Arc_Energy_J': round(sim.arc_energy_effective, 1),
        'Contact_R_Loose_ohm': round(sim.R_loose, 6),
        'Contact_R_Oxide_ohm': round(sim.R_oxide, 6),
        'Contact_R_Carbon_ohm': round(sim.R_carbon, 6),
        'Peak_Arc_Power_W': round(sim.peak_arc_power, 1),
        'Peak_Fault_Voltage_V': round(sim.peak_fault_voltage, 1),
        'Peak_Core_T_C': round(sim.peak_core_temp, 1),
        'Peak_T_Air_Zone_C': round(sim.peak_T_air_zone, 1),
        'Peak_T_Solid_NonAir_C': round(sim.peak_T_solid_non_air, 1),
        'Peak_Fire_HRR_W': round(sim.peak_fire_hrr, 1),
        'Nodes_Ignited': int(sum(sim.node_ignited.values())),
        'Secondary_Ignition_Causes': ';'.join(
            f'{node}:{sim.node_ignition_cause[node]}'
            for node in sim.nodes
            if node != sim.f_node and sim.node_ignited[node]
        ),
        'Max_Thermal_Spread_Duration_s': round(
            max(
                (duration for node, duration in sim.thermal_spread_duration.items() if node != sim.f_node),
                default=0.0,
            ),
            3,
        ),
        'AC_Zero_Crossings': int(sim.ac_zero_cross_total),
        'AC_Reignitions': int(sim.ac_reignition_total),
        'Zero_Crossings': int(sim.ac_zero_cross_total),
        'Reignitions': int(sim.ac_reignition_total),
        'Arc_Quenched_No_Ignition': bool(sim.arc_quenched and t_stage3 is None),
        'Capability_Tag': '；'.join(capability_tags),
        'Limitation_Tag': '；'.join(limitation_tags),
    }

    # Pre-ignition fair comparison metrics
    if history_df is not None and not history_df.empty and t_stage3 is not None:
        pre_ig = history_df[history_df['Time'] <= t_stage3]
        if not pre_ig.empty:
            t_core_col = f"{sim.f_node}_T_Core"
            c_col = f"{sim.f_node}_C"
            if t_core_col in pre_ig.columns:
                result['Max_PreIg_T_Fault_C'] = round(float(pre_ig[t_core_col].max()), 2)
                # AUC: integral of (T - T_ig) above threshold
                t_above = (pre_ig[t_core_col] - sim.T_ig).clip(lower=0.0)
                dt = pre_ig['Time'].diff().fillna(sim.dt_record)
                result['AUC_PreIg_T_Fault'] = round(float((t_above * dt).sum()), 2)
            if c_col in pre_ig.columns:
                result['Max_PreIg_C_Fault'] = round(float(pre_ig[c_col].max()), 2)
                dt = pre_ig['Time'].diff().fillna(sim.dt_record)
                result['AUC_PreIg_C_Fault'] = round(float((pre_ig[c_col] * dt).sum()), 2)
            # Fixed time-point snapshots
            for t_fixed in [400.0, 500.0]:
                row_at_t = pre_ig[pre_ig['Time'] >= t_fixed]
                if not row_at_t.empty:
                    if t_core_col in row_at_t.columns:
                        result[f'T_Fault_at_{int(t_fixed)}s'] = round(float(row_at_t.iloc[0][t_core_col]), 2)
                    if c_col in row_at_t.columns:
                        result[f'C_Fault_at_{int(t_fixed)}s'] = round(float(row_at_t.iloc[0][c_col]), 2)
    else:
        result['Max_PreIg_T_Fault_C'] = None
        result['Max_PreIg_C_Fault'] = None
        result['AUC_PreIg_T_Fault'] = None
        result['AUC_PreIg_C_Fault'] = None
        result['T_Fault_at_400s'] = None
        result['C_Fault_at_400s'] = None
        result['T_Fault_at_500s'] = None
        result['C_Fault_at_500s'] = None

    snapshot = getattr(sim, 'stage15_trigger_snapshot', None)

    if snapshot is not None:
        result.update({
            'Stage15_Trigger_Time_s': snapshot.get('time_s'),
            'Stage15_Trigger_Source': snapshot.get('source'),

            'Stage15_Trigger_Contact_Power': snapshot.get('contact_power_est'),
            'Stage15_Trigger_T_eff': snapshot.get('T_eff'),
            'Stage15_Trigger_C_Raw': snapshot.get('C_raw_fault'),
            'Stage15_Trigger_Volatile_Pool_g': snapshot.get('volatile_pool_fault_g'),
            'Stage15_Trigger_VolatileRelease_Exposure': snapshot.get('volatile_release_exposure'),
            'Stage15_Trigger_Pyrolyzed_Mass_g': snapshot.get('pyrolyzed_mass_fault_g'),
            'Stage15_Trigger_Gas_Exposure': snapshot.get('gas_exposure_fault'),

            'Stage15_Trigger_C_Hard_Ready': snapshot.get('Gate15_C_Hard_Ready'),
            'Stage15_Trigger_C_Soft_Ready': snapshot.get('Gate15_C_Soft_Ready'),
            'Stage15_Trigger_Volatile_Ready': snapshot.get('Gate15_Volatile_Ready'),
            'Stage15_Trigger_VolatileRelease_Ready': snapshot.get('Gate15_VolatileRelease_Ready'),
            'Stage15_Trigger_PyroMass_Ready': snapshot.get('Gate15_PyroMass_Ready'),
            'Stage15_Trigger_GasExposure_Ready': snapshot.get('Gate15_GasExposure_Ready'),
            'Stage15_Trigger_PyroAux_Ready': snapshot.get('Gate15_PyroAux_Ready'),
        })
    else:
        result.update({
            'Stage15_Trigger_Time_s': None,
            'Stage15_Trigger_Source': None,
        })

    stage2_snapshot = getattr(sim, "stage2_trigger_snapshot", None)

    if stage2_snapshot is not None:
        result.update({
            "Stage2_Trigger_Time_s": stage2_snapshot.get("time_s"),
            "Stage2_Trigger_Carb_Track": stage2_snapshot.get("carb_track"),
            "Stage2_Trigger_Arc_Energy_Increment": stage2_snapshot.get("arc_energy_since_stage15"),
            "Stage2_Trigger_Duration_s": stage2_snapshot.get("stage15_duration"),
            "Stage2_Carb_Ready_at_Stage2": stage2_snapshot.get("Stage2_Carb_Ready_at_Stage2"),
            "Stage2_ArcEnergy_Ready_at_Stage2": stage2_snapshot.get("Stage2_ArcEnergy_Ready_at_Stage2"),
            "Stage2_Duration_Ready_at_Stage2": stage2_snapshot.get("Stage2_Duration_Ready_at_Stage2"),
        })
    else:
        result.update({
            "Stage2_Trigger_Time_s": None,
            "Stage2_Trigger_Carb_Track": None,
            "Stage2_Trigger_Arc_Energy_Increment": None,
            "Stage2_Trigger_Duration_s": None,
            "Stage2_Carb_Ready_at_Stage2": None,
            "Stage2_ArcEnergy_Ready_at_Stage2": None,
            "Stage2_Duration_Ready_at_Stage2": None,
        })

    fault = sim.f_node

    extra_diag = {
        'Fault_C_Raw_Final': round(sim.states[fault][1], 4),
        'Fault_Volatile_Pool_Final_g': round(sim.volatile_pool[fault], 5),
        'Fault_Char_Fuel_Final_g': round(sim.char_fuel_mass[fault], 5),
        'Fault_Pyrolyzed_Mass_Final_g': round(sim.pyrolyzed_mass[fault], 5),
        'Fault_Remaining_Fuel_Final_g': round(sim.remaining_fuel[fault], 5),
        'Fault_Remaining_Fuel_at_Stage3_g': (
            round(sim.fault_remaining_fuel_at_stage3, 5)
            if sim.fault_remaining_fuel_at_stage3 is not None
            else None
        ),
        'Fault_Volatile_Pool_at_Stage3_g': (
            round(sim.fault_volatile_pool_at_stage3, 5)
            if getattr(sim, 'fault_volatile_pool_at_stage3', None) is not None
            else None
        ),
        'Fault_Char_Fuel_at_Stage3_g': (
            round(sim.fault_char_fuel_at_stage3, 5)
            if getattr(sim, 'fault_char_fuel_at_stage3', None) is not None
            else None
        ),
        'Fault_Burnable_Mass_at_Stage3_g': (
            round(sim.fault_burnable_mass_at_stage3, 5)
            if getattr(sim, 'fault_burnable_mass_at_stage3', None) is not None
            else None
        ),
        'Fuel_Depletion_Time_s': (
            round(sim.fuel_depletion_time, 3)
            if sim.fuel_depletion_time is not None
            else None
        ),
        'Solid_Fuel_Depletion_Time_s': (
            round(sim.solid_fuel_depletion_time, 3)
            if getattr(sim, 'solid_fuel_depletion_time', None) is not None
            else None
        ),
        'Burnable_Depletion_Time_s': (
            round(sim.burnable_depletion_time, 3)
            if getattr(sim, 'burnable_depletion_time', None) is not None
            else None
        ),
        'Arc_Without_Sustained_Fire': bool(getattr(sim, 'arc_without_sustained_fire', False)),
        'No_Burnable_Fire_Attempt_Time_s': (
            round(sim.no_burnable_fire_attempt_time, 3)
            if getattr(sim, 'no_burnable_fire_attempt_time', None) is not None
            else None
        ),
        'Volatile_Burn_Weight': getattr(sim, 'volatile_burn_weight', 0.8),
        'Char_Burn_Weight': getattr(sim, 'char_burn_weight', 0.50),
        'Char_Ignition_Weight': getattr(sim, 'char_ignition_weight', 0.08),
        'Min_Burnable_Mass_Abs': getattr(sim, 'min_burnable_mass_abs', 0.15),
        'Min_Burnable_Mass_Frac': getattr(sim, 'min_burnable_mass_frac', 0.02),
        'Fault_Gas_Exposure_Final': round(sim.gas_exposure[fault], 4),

        'Fault_Volatile_Generated_Final_g': round(sim.volatile_generated_mass[fault], 5),
        'Fault_Volatile_Pool_Peak_g': round(sim.volatile_pool_peak[fault], 5),
        'Fault_Volatile_Release_Exposure_Final_g': round(sim.volatile_release_exposure[fault], 5),

        'Gas_Stage_Threshold': sim.gas_stage_threshold,
        'Gas_Stage_Soft_Threshold': sim.gas_stage_soft_threshold,
        'Volatile_Stage_Threshold_g': sim.volatile_stage_threshold,
        'Pyro_Mass_Stage_Threshold_g': sim.pyro_mass_stage_threshold,
        'Gas_Exposure_Base': sim.gas_exposure_base,
        'Gas_Exposure_Stage_Threshold': sim.gas_exposure_stage_threshold,
        'Volatile_Release_Exposure_Threshold_g': sim.volatile_release_exposure_threshold,

        'Min_Stage15_Duration_s': sim.min_stage15_duration,
        'Stage15_to_Stage2_Arc_Energy_Threshold_J': sim.stage15_to_stage2_arc_energy,

        'Carb_Track_Final': round(sim.carb_track[sim.f_node], 5),
        'Carb_Track_At_Stage15': (
            round(sim.carb_track_at_stage15, 5)
            if sim.carb_track_at_stage15 is not None else None
        ),
        'Stage1_Carb_Cap': sim.stage1_carb_cap,
        'Stage1_Carb_Gain': sim.stage1_carb_gain,
        'Stage15_Carb_Gain': sim.stage15_carb_gain,
        'Stage15_Arc_Carb_Gain': sim.stage15_arc_carb_gain,
        'Stage2_Carb_Threshold': sim.stage2_carb_threshold,

        # ---------- Stage 2 final readiness ----------
        'Stage2_Carb_Ready_Final': int(
            sim.carb_track[sim.f_node] >= sim.stage2_carb_threshold
        ),
        'Stage15_Actual_Duration_s': round(
            (t_stage2 - t_stage1_5 if t_stage2 is not None else 0.0)
            if t_stage1_5 is not None else 0.0,
            4,
        ),
        'Stage2_Duration_Ready_Final': int(
            (t_stage1_5 is not None)
            and (
                (t_stage2 - t_stage1_5 if t_stage2 is not None else 0.0)
                >= sim.min_stage15_duration
            )
        ),

        # ---------- Stage 3 validity ----------
        'Fault_Burnable_Mass_Final_g': round(get_burnable_mass(sim, fault), 5),
        'Min_Burnable_Mass_Fault_g': round(get_min_burnable_mass(sim, fault), 5),
    }

    # ArcEnergy ready final
    if sim.carb_track_at_stage15 is not None:
        arc_energy_since_stage15_final = sim.arc_energy_effective - sim.arc_energy_at_stage15
    else:
        arc_energy_since_stage15_final = 0.0
    extra_diag['Stage2_ArcEnergy_Ready_Final'] = int(
        arc_energy_since_stage15_final >= sim.stage15_to_stage2_arc_energy
    )

    # Diagnostics issues
    issues = []
    if t_stage3 is not None and sim.peak_fire_hrr <= 1e-6:
        issues.append("Stage 3 was triggered but peak_fire_hrr is zero; invalid sustained fire.")
    extra_diag['Diagnostic_Issues'] = "; ".join(issues) if issues else None

    result.update(extra_diag)

    if history_df is not None and not history_df.empty:
        if 'Fault_C_Raw' in history_df.columns:
            result['Fault_C_Raw_Peak'] = round(float(history_df['Fault_C_Raw'].max()), 4)
        if 'Fault_Volatile_Pool_g' in history_df.columns:
            result['Fault_Volatile_Pool_Peak_g'] = round(float(history_df['Fault_Volatile_Pool_g'].max()), 5)
        if 'Fault_Gas_Exposure' in history_df.columns:
            result['Fault_Gas_Exposure_Peak'] = round(float(history_df['Fault_Gas_Exposure'].max()), 4)
        for col in [
            'Fault_Volatile_Release_Exposure_g',
            'Stage15_Duration_s',
            'Stage15_Carb_Increment',
            'Stage15_Arc_Energy_Increment',
            'Carb_Track_Fault',
        ]:
            if col in history_df.columns:
                result[f'{col}_Max'] = round(float(history_df[col].max()), 5)

    return result
