"""Lightweight regression checks for the revised cabinet fault physics model."""
from __future__ import annotations

from multiphysics_accident_model.core.state import PIGAT_Realistic_DigitalTwin_V3
from multiphysics_accident_model.physics.gas_smoke_model import update_gas_smoke_step
from multiphysics_accident_model.physics.ignition_spread import (
    compute_ignition_probability,
    get_burnable_mass,
    get_ignitable_mass,
    update_combustion,
    update_secondary_spread,
)


def make_sim(seed=7, vent_state="normal", service_years=0.0):
    return PIGAT_Realistic_DigitalTwin_V3(
        init_current=30.0,
        fault_terminal="KM1_DC",
        service_years=service_years,
        seed=seed,
        vent_state=vent_state,
    )


def check_thermal_surface_spread():
    sim = make_sim()
    source = sim.f_node
    neighbor = "KM1_DC_OUT"
    sim.stage = 3.0
    sim.node_ignited[source] = True
    sim.node_hrr[source] = 1200.0
    sim.states[neighbor][0] = 640.0
    sim.states[neighbor][1] = 0.0
    heat = {node: 0.0 for node in sim.nodes}

    for step in range(410):
        update_secondary_spread(sim, heat, step * sim.dt_record)

    assert sim.node_ignited[neighbor]
    assert sim.node_ignition_cause[neighbor] == "thermal_surface_spread"


def check_char_residue_is_burnable():
    sim = make_sim(service_years=30.0)
    node = sim.f_node
    sim.remaining_fuel[node] = 1.0
    before = sim.char_fuel_mass[node]
    next_t = {name: sim.states[name][0] for name in sim.nodes}
    pyro_rates = {name: 0.0 for name in sim.nodes}
    gas_rates = {name: 0.0 for name in sim.nodes}
    char_rates = {name: 0.0 for name in sim.nodes}
    pyro_rates[node] = 2.0

    update_gas_smoke_step(sim, next_t, pyro_rates, gas_rates, char_rates)
    assert sim.char_fuel_mass[node] > before

    sim.remaining_fuel[node] = 0.0
    sim.volatile_pool[node] = 0.0
    assert get_burnable_mass(sim, node) > 0.0
    assert get_ignitable_mass(sim, node) < get_burnable_mass(sim, node)
    sim.node_ignited[node] = True
    char_before_burn = sim.char_fuel_mass[node]
    update_combustion(sim)
    assert sim.node_hrr[node] > 0.0
    assert sim.char_fuel_mass[node] < char_before_burn


def check_char_residue_does_not_force_primary_flame():
    sim = make_sim(service_years=30.0)
    node = sim.f_node
    sim.remaining_fuel[node] = 0.0
    sim.volatile_pool[node] = 0.0
    sim.char_fuel_mass[node] = 4.4
    assert get_ignitable_mass(sim, node) < sim.min_burnable_mass_frac * sim.fuel_mass_total[node]


def check_ventilation_affects_ignition():
    off = make_sim(vent_state="off")
    forced = make_sim(vent_state="forced")
    for sim in (off, forced):
        node = sim.f_node
        sim.states[node][0] = 300.0
        sim.states[node][1] = 12.0
        sim.arc_energy_effective = 600.0
    assert off.vent_oxygen_scale < forced.vent_oxygen_scale
    assert compute_ignition_probability(off, off.f_node) < compute_ignition_probability(forced, forced.f_node)


def check_aging_separates_fault_risk_from_flame_response():
    fresh = make_sim(service_years=0.0)
    aged = make_sim(service_years=30.0)
    assert aged.R_metal > fresh.R_metal
    assert aged.T_ig > fresh.T_ig
    assert aged.gas_ignition_threshold > fresh.gas_ignition_threshold
    assert aged.aging_profile.Ea > fresh.aging_profile.Ea
    assert aged.aging_profile.combustion_rate_multiplier < fresh.aging_profile.combustion_rate_multiplier


def check_arc_seed_reproducibility():
    left = make_sim(seed=123)
    right = make_sim(seed=123)
    left.arc_gap_mm = right.arc_gap_mm = 0.4
    left.carb_track[left.f_node] = right.carb_track[right.f_node] = 0.5
    result_left = left._compute_dc_arc_branch(4.0, intermittent=True)
    result_right = right._compute_dc_arc_branch(4.0, intermittent=True)
    assert result_left == result_right


def check_short_run_diagnostics():
    sim = make_sim(seed=321)
    node = sim.f_node
    sim.stage = 2.0
    sim.arc_duration = sim.min_ignition_duration
    sim.ignition_drive[node] = sim.ignition_drive_threshold
    sim.states[node][0] = 420.0
    sim.states[node][1] = 40.0
    sim.arc_energy_effective = 1200.0
    sim.post_ignition_observation = 0.04
    _, _, _, t3, _, diagnostics = sim.run(record_history=False)
    assert t3 is not None
    assert diagnostics["Nodes_Ignited"] >= 1
    assert "Fault_Char_Fuel_Final_g" in diagnostics
    assert "Secondary_Ignition_Causes" in diagnostics


def main():
    checks = [
        check_thermal_surface_spread,
        check_char_residue_is_burnable,
        check_char_residue_does_not_force_primary_flame,
        check_ventilation_affects_ignition,
        check_aging_separates_fault_risk_from_flame_response,
        check_arc_seed_reproducibility,
        check_short_run_diagnostics,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
