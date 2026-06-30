import numpy as np

from .pyrolysis_model import compute_pyrolysis_rate

SIGMA = 5.67e-8
WALL_EMISSIVITY = 0.85
C_P_AIR = 1005.0
BASE_AIR_FLOW_KG_S = 0.06
BUOYANCY_GAIN = 0.6
BUOYANCY_SCALE_K = 25.0
MAX_TEMP_C_FOR_RADIATION = 1500.0
DT_CLIP_K_PER_S = 8000.0
AIR_ZONE_FIRE_ABS = 0.05
AIR_ZONE_WALL_LOSS_K = 4.0
AIR_ZONE_VENT_LOSS_K = 6.0

AIR_ZONE_TYPE = 'air_zone'
BOUNDARY_NODE_TYPES = ('cabinet_boundary', 'ventilation')


def _buoyancy_factor(delta_T_abs):
    return 1.0 + BUOYANCY_GAIN * np.tanh(max(delta_T_abs, 0.0) / BUOYANCY_SCALE_K)


def _wall_temperature(sim, T):
    boundary_nodes = sim.cabinet_boundary_nodes
    if not boundary_nodes:
        return sim.ambient_temp
    return sum(T[n] for n in boundary_nodes) / len(boundary_nodes)


def _air_zone_for(sim, node, T):
    z = sim.solid_air_zone.get(node)
    if z is None:
        return sim.ambient_temp
    return T[z]


def _radiation_heat(eps, A, T_self_c, T_other_c):
    t_self = min(T_self_c, MAX_TEMP_C_FOR_RADIATION) + 273.15
    t_other = T_other_c + 273.15
    return eps * SIGMA * A * (t_self ** 4 - t_other ** 4)


def _solid_node_dt(sim, node, T, p_elec, p_fire, T_wall):
    is_boundary = sim.node_type[node] in BOUNDARY_NODE_TYPES
    Q_in = p_elec[node] + p_fire[node]

    q_cond_in = 0.0
    for other, k_th in sim.solid_thermal_neighbors.get(node, []):
        q_cond_in += k_th * (T[other] - T[node])

    if is_boundary:
        T_air_local = sim.ambient_temp
        T_rad = sim.ambient_temp
    else:
        T_air_local = _air_zone_for(sim, node, T)
        T_rad = T_wall
    q_conv = sim.conv_hA[node] * (T[node] - T_air_local)
    q_rad = _radiation_heat(WALL_EMISSIVITY, sim.surface_area[node], T[node], T_rad)

    return (Q_in + q_cond_in - q_conv - q_rad) / max(sim.C_th[node], 1e-6)


def _air_zone_dt(sim, zone, T, p_fire, T_wall):
    Q = 0.0
    for member in sim.air_zone_members.get(zone, []):
        Q += sim.conv_hA[member] * (T[member] - T[zone])

    vent_scale = getattr(sim, 'vent_flow_multiplier', 1.0)
    for other_zone, a_air, _flow_dir in sim.air_zone_air_neighbors.get(zone, []):
        delta = T[other_zone] - T[zone]
        m_dot = BASE_AIR_FLOW_KG_S * a_air * _buoyancy_factor(abs(delta)) * vent_scale
        Q += m_dot * C_P_AIR * delta

    a_out = sim.air_zone_outlet_a.get(zone, 0.0)
    if a_out > 0.0:
        delta_out = T[zone] - sim.ambient_temp
        m_out = BASE_AIR_FLOW_KG_S * a_out * _buoyancy_factor(abs(delta_out)) * vent_scale
        Q -= m_out * C_P_AIR * delta_out

    for wall_node, k_th in sim.air_zone_wall_neighbors.get(zone, []):
        Q += k_th * (T[wall_node] - T[zone])

    k_wall = getattr(sim, 'air_zone_wall_loss_coeff', AIR_ZONE_WALL_LOSS_K)
    Q -= k_wall * (T[zone] - T_wall)

    k_vent_heat = getattr(sim, 'air_zone_vent_loss_coeff', AIR_ZONE_VENT_LOSS_K)
    heat_removal_multiplier = getattr(sim, 'heat_removal_multiplier',
                                       getattr(sim, 'vent_heat_removal_scale', 1.0))
    Q -= k_vent_heat * heat_removal_multiplier * (T[zone] - sim.ambient_temp)

    Q += AIR_ZONE_FIRE_ABS * p_fire[zone]
    return Q / max(sim.C_th[zone], 1e-6)


def compute_thermal_step(sim, inst_cache):
    T = {n: sim.states[n][0] for n in sim.nodes}
    p_elec = {n: float(np.mean(inst_cache[n]['P'])) for n in sim.nodes}
    p_fire = {n: sim._compute_incident_fire_heat(n) for n in sim.nodes}

    T_wall = _wall_temperature(sim, T)

    next_T = {}
    pyro_rates = {}
    for n in sim.nodes:
        if sim.node_type[n] == AIR_ZONE_TYPE:
            dT = _air_zone_dt(sim, n, T, p_fire, T_wall)
        else:
            dT = _solid_node_dt(sim, n, T, p_elec, p_fire, T_wall)
        dT = float(np.clip(dT, -DT_CLIP_K_PER_S, DT_CLIP_K_PER_S))
        next_T[n] = T[n] + dT * sim.dt_record
        pyro_rates[n] = compute_pyrolysis_rate(sim, n, p_elec[n])

    return next_T, pyro_rates, p_fire
