#!/usr/bin/env python
"""Run a single calibration case with specified parameters."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multiphysics_accident_model.core.state import PIGAT_Realistic_DigitalTwin_V3

# Calibration case parameters
fault_terminal = 'KM1_DC'
current_level = 20.0
service_years = 10.0
vent_state = 'normal'
door_state = 'closed'
severity = 0.75
current_profile = 'constant'
ambient_temp = 26.0
seed = 42

print(f"{'='*70}")
print(f"Running calibration case:")
print(f"  fault_terminal = {fault_terminal}")
print(f"  current_level  = {current_level} A")
print(f"  service_years  = {service_years}")
print(f"  vent_state     = {vent_state}")
print(f"  door_state     = {door_state}")
print(f"  severity       = {severity}")
print(f"  current_profile = {current_profile}")
print(f"  ambient_temp   = {ambient_temp} C")
print(f"  seed           = {seed}")
print(f"{'='*70}\n")

sim = PIGAT_Realistic_DigitalTwin_V3(
    init_current=current_level,
    fault_terminal=fault_terminal,
    case_id=None,
    service_days=30,
    service_years=service_years,
    seed=seed,
    vent_state=vent_state,
)

# Override specific parameters after initialization
sim.ambient_temp = ambient_temp
sim.door_state = door_state
sim.current_profile_name = current_profile

# Override fault profile severity
sim.fault_profile.severity = severity

# Also override the case config values that feed into other parameters
for node in sim.nodes:
    sim.states[node][0] = ambient_temp
    sim.sensor_T[node] = ambient_temp
sim.peak_core_temp = ambient_temp

print(f"Case ID        : {sim.case_id}")
print(f"Fault terminal : {sim.f_node}")
print(f"Supply mode    : {sim.supply_mode}")
print(f"Circuit ID     : {sim.circuit_id}")
print(f"Fuel mass (g)  : {sim.fuel_mass_total[sim.f_node]:.3f}")
print(f"T_pyro         : {sim.T_pyro}")
print(f"T_ig           : {sim.T_ig}")
print(f"P_crit_contact : {sim.P_crit_contact}")
print(f"gas_stage_threshold : {sim.gas_stage_threshold}")
print(f"gas_stage_soft_threshold : {sim.gas_stage_soft_threshold}")
print(f"volatile_stage_threshold : {sim.volatile_stage_threshold}")
print(f"pyro_mass_stage_threshold : {sim.pyro_mass_stage_threshold}")
print(f"gas_exposure_stage_threshold : {sim.gas_exposure_stage_threshold}")
print(f"gas_exposure_base : {sim.gas_exposure_base}")
print(f"volatile_release_exposure_threshold : {sim.volatile_release_exposure_threshold}")
print(f"pyro_volatile_yield : {sim.pyro_volatile_yield}")
print(f"pyro_char_yield : {sim.pyro_char_yield}")
print(f"volatile_release_base : {sim.volatile_release_base}")
print(f"volatile_release_temp_scale : {sim.volatile_release_temp_scale}")
print(f"min_stage15_duration : {sim.min_stage15_duration}")
print(f"stage1_carb_cap : {sim.stage1_carb_cap}")
print(f"stage1_carb_gain : {sim.stage1_carb_gain}")
print(f"stage15_carb_gain : {sim.stage15_carb_gain}")
print(f"stage15_arc_carb_gain : {sim.stage15_arc_carb_gain}")
print(f"stage2_carb_threshold : {sim.stage2_carb_threshold}")
print(f"stage15_to_stage2_arc_energy : {sim.stage15_to_stage2_arc_energy}")
print(f"\n{'='*70}")
print("Running case model...")
print(f"{'='*70}\n")

df, t1_5, t2, t3, q_at_ig, diag = sim.run()

status = '未起火' if t3 is None else '起火'
t1_5_str = 'None' if t1_5 is None else f'{t1_5:.3f}'
t2_str = 'None' if t2 is None else f'{t2:.3f}'
t3_str = 'None' if t3 is None else f'{t3:.3f}'

print(f"\n{'='*70}")
print("SIMULATION RESULTS")
print(f"{'='*70}")
print(f"Stage 1.5 (微弧) 时间 : {t1_5_str} s")
print(f"Stage 2.0 (稳弧) 时间 : {t2_str} s")
print(f"Stage 3.0 (起火) 时间 : {t3_str} s")
print(f"引燃时弧能           : {q_at_ig:.1f} J" if q_at_ig else "引燃时弧能           : N/A")
print(f"峰值弧功率           : {diag['Peak_Arc_Power_W']:.1f} W")
print(f"峰值核心温度         : {diag['Peak_Core_T_C']:.1f} C")
print(f"峰值火源HRR          : {diag['Peak_Fire_HRR_W']:.1f} W")
print(f"起火节点数           : {diag['Nodes_Ignited']}")
print(f"最终状态             : {status}")
print(f"弧是否熄灭           : {diag['Arc_Quenched_No_Ignition']}")

print(f"\n{'='*70}")
print("FAULT NODE DIAGNOSTICS")
print(f"{'='*70}")
print(f"C_Raw (final)        : {diag.get('Fault_C_Raw_Final', 'N/A')}")
print(f"Volatile Pool (final): {diag.get('Fault_Volatile_Pool_Final_g', 'N/A')} g")
print(f"Pyrolyzed Mass (final): {diag.get('Fault_Pyrolyzed_Mass_Final_g', 'N/A')} g")
print(f"Remaining Fuel (final): {diag.get('Fault_Remaining_Fuel_Final_g', 'N/A')} g")
print(f"Gas Exposure (final) : {diag.get('Fault_Gas_Exposure_Final', 'N/A')}")
print(f"C_Raw (peak)         : {diag.get('Fault_C_Raw_Peak', 'N/A')}")
print(f"Volatile Pool (peak) : {diag.get('Fault_Volatile_Pool_Peak_g', 'N/A')} g")
print(f"Gas Exposure (peak)  : {diag.get('Fault_Gas_Exposure_Peak', 'N/A')}")

print(f"\n{'='*70}")
print("STAGE GATE READINESS (final)")
print(f"{'='*70}")
print(f"Gas Stage Threshold      : {diag.get('Gas_Stage_Threshold', 'N/A')}")
print(f"Gas Stage Soft Threshold : {diag.get('Gas_Stage_Soft_Threshold', 'N/A')}")
print(f"Volatile Stage Threshold : {diag.get('Volatile_Stage_Threshold_g', 'N/A')} g")
print(f"Pyro Mass Stage Threshold: {diag.get('Pyro_Mass_Stage_Threshold_g', 'N/A')} g")
print(f"Gas Exposure Threshold   : {diag.get('Gas_Exposure_Stage_Threshold', 'N/A')}")
print(f"Gas Exposure Base        : {diag.get('Gas_Exposure_Base', 'N/A')}")
print(f"Volatile Release Exp Thr : {diag.get('Volatile_Release_Exposure_Threshold_g', 'N/A')} g")

print(f"\n{'='*70}")
print("STAGE 1.5 TRIGGER SNAPSHOT (from diagnostics)")
print(f"{'='*70}")
for k in sorted(diag.keys()):
    if k.startswith('Stage15_Trigger_'):
        print(f"  {k:40s}: {diag[k]}")

print(f"\n{'='*70}")
print("STAGE 1.5 -> 2.0 TRANSITION DIAGNOSTICS")
print(f"{'='*70}")
print(f"  Min Stage15 Duration       : {diag.get('Min_Stage15_Duration_s', 'N/A')} s")
print(f"  Arc Energy Threshold       : {diag.get('Stage15_to_Stage2_Arc_Energy_Threshold_J', 'N/A')} J")
print(f"  Stage2 Carb Threshold      : {diag.get('Stage2_Carb_Threshold', 'N/A')}")
print(f"  carb_track_at_stage15      : {sim.carb_track_at_stage15}")
print(f"  arc_energy_at_stage15      : {sim.arc_energy_at_stage15:.3f} J")
print(f"  stage15_start_time         : {sim.stage15_start_time}")
if sim.stage15_start_time is not None and t2 is not None:
    elapsed = t2 - sim.stage15_start_time
    print(f"  actual_stage15_duration    : {elapsed:.3f} s")
    carb_incr = sim.carb_track[sim.f_node] - (sim.carb_track_at_stage15 or 0)
    arc_incr = diag.get('Arc_Energy', 0) - sim.arc_energy_at_stage15
    print(f"  carb_increment             : {carb_incr:.4f}")
    print(f"  arc_energy_increment       : {arc_incr:.3f} J")
    print(f"  Stage15_Duration_Max       : {diag.get('Stage15_Duration_s_Max', 'N/A')}")
    print(f"  Stage15_Carb_Increment_Max : {diag.get('Stage15_Carb_Increment_Max', 'N/A')}")
    print(f"  Stage15_Arc_Energy_Inc_Max : {diag.get('Stage15_Arc_Energy_Increment_Max', 'N/A')}")
else:
    print(f"  Stage 2.0 was not reached.")

print(f"\n{'='*70}")
print("VOLATILE DIAGNOSTICS")
print(f"{'='*70}")
print(f"  Volatile Generated (final) : {diag.get('Fault_Volatile_Generated_Final_g', 'N/A')} g")
print(f"  Volatile Pool Peak         : {diag.get('Fault_Volatile_Pool_Peak_g', 'N/A')} g")
print(f"  Volatile Release Exp (final): {diag.get('Fault_Volatile_Release_Exposure_Final_g', 'N/A')} g")
print(f"  Fault_Volatile_Release_Exposure_g_Max: {diag.get('Fault_Volatile_Release_Exposure_g_Max', 'N/A')}")

print(f"\n{'='*70}")
print("SAMPLE COUNTS")
print(f"{'='*70}")
print(f"Stage 1.0 samples : {sim.stage_sample_counts[1.0]}")
print(f"Stage 1.5 samples : {sim.stage_sample_counts[1.5]}")
print(f"Stage 2.0 samples : {sim.stage_sample_counts[2.0]}")
print(f"Stage 3.0 samples : {sim.stage_sample_counts.get(3.0, 0)}")

# Save history to CSV for inspection
output_file = 'calibration_case_history.csv'
df.to_csv(output_file, index=False)
print(f"\nHistory saved to: {output_file}")
print(f"Total history rows: {len(df)}")

# Print some key time-series snapshots
if not df.empty:
    print(f"\n{'='*70}")
    print("KEY TIMESERIES SNAPSHOTS")
    print(f"{'='*70}")
    key_cols = ['Time', 'Stage', 'Fault_Resistance', 'Gas_Concentration',
                'Fault_C_Raw', 'Fault_Volatile_Pool_g', 'Fault_Pyrolyzed_Mass_g',
                'Fault_Gas_Exposure', 'Gate15_C_Raw_Ready', 'Gate15_Volatile_Ready',
                'Gate15_PyroMass_Ready', 'Gate15_GasExposure_Ready',
                'Arc_Power', 'Arc_Energy',
                f'{fault_terminal}_T_Core', f'{fault_terminal}_C']
    available_cols = [c for c in key_cols if c in df.columns]

    # Print at interesting time points
    for t_target in [5.0, 50.0, 100.0, 200.0, 400.0, 600.0]:
        rows_near = df[df['Time'] >= t_target]
        if not rows_near.empty:
            row = rows_near.iloc[0]
            print(f"\n  t = {row['Time']:.1f}s (requested {t_target}s):")
            for col in available_cols:
                print(f"    {col:30s}: {row[col]}")
        else:
            print(f"\n  t = {t_target}s: no data (sim ended earlier)")
