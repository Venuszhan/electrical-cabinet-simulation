import numpy as np


def compute_pyrolysis_rate(sim, node, p_elec):
    t_k = max(sim.states[node][0] + 273.15, 273.15)
    fuel_factor = max(sim.remaining_fuel[node] / max(sim.fuel_mass_total[node], 1e-6), 0.0)
    electrical_factor = 0.35 + 0.65 * min(1.0, p_elec / 150.0)
    aging_profile = sim.aging_profiles[node]
    a_pyro = sim.A_pyro_base * aging_profile.pyrolysis_rate_multiplier
    return a_pyro * np.exp(-aging_profile.Ea / (8.314 * t_k)) * fuel_factor * sim.fuel_mass_total[node] * electrical_factor


def compute_gas_source_rate(sim, node, pyro_rate):
    """气体源项：α_i * ṁ_pyro。α 为热解产气系数，随材料类型和老化变化。"""
    if pyro_rate <= 0.0:
        return 0.0
    # 基础产气系数，不同节点类型产气率不同
    base_yield = {
        'switch': 18.0,
        'breaker': 18.0,
        'contactor': 22.0,
        'contactor_terminal': 24.0,
        'protector': 15.0,
        'relay': 16.0,
        'terminal': 25.0,
        'connector': 23.0,
        'harness': 20.0,
        'ground': 5.0,
        'air_zone': 0.0,
        'cabinet_boundary': 0.0,
        'ventilation': 0.0,
    }.get(sim.node_type[node], 18.0)
    aging_boost = 1.0 + 0.35 * sim.aging_profiles[node].eta_insulation
    alpha = base_yield * aging_boost
    return alpha * pyro_rate
