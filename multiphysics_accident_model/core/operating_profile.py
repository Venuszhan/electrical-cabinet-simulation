import numpy as np


VENT_STATE_MULTIPLIER = {
    'off': 0.45,
    'reduced': 0.70,
    'normal': 1.00,
    'forced': 1.35,
}

# 验证阶段使用的三维通风参数:
#   heat_removal - 对流散热倍率(乘到 air_zone / ventilation 节点的 conv_hA)
#   gas_outflow  - 气体外流倍率(决定 vent_flow_multiplier、气体外漏率)
#   oxygen       - 机柜内氧气可用性倍率(乘到燃烧 oxygen_factor)
# 仅当 vent_state 命中本字典时启用三维通风模型;否则按 VENT_STATE_MULTIPLIER 单维度处理。
VENT_STATE_PARAMS = {
    'normal':     {'heat_removal': 1.00, 'gas_outflow': 1.00, 'oxygen': 1.00},
    'reduced':    {'heat_removal': 0.70, 'gas_outflow': 0.70, 'oxygen': 0.92},
    'off':        {'heat_removal': 0.45, 'gas_outflow': 0.45, 'oxygen': 0.78},
    'blocked_30': {'heat_removal': 0.70, 'gas_outflow': 0.70, 'oxygen': 0.95},
    'blocked_60': {'heat_removal': 0.40, 'gas_outflow': 0.40, 'oxygen': 0.85},
    'fan_failed': {'heat_removal': 0.45, 'gas_outflow': 0.45, 'oxygen': 0.90},
    'forced':     {'heat_removal': 1.35, 'gas_outflow': 1.35, 'oxygen': 1.08},
}

DOOR_STATE_MULTIPLIER = {
    'closed': 1.00,
    'ajar': 1.25,
    'open': 1.65,
}


class CurrentProfile:
    def __init__(self, profile_type, base_current, t_fault_start, rng=None):
        self.profile_type = profile_type
        self.base_current = float(base_current)
        self.t_fault_start = float(t_fault_start)
        self.rng = rng
        self.phase = 0.0 if rng is None else float(rng.uniform(0.0, 2.0 * np.pi))

    def value(self, t):
        if self.profile_type == 'constant':
            scale = 1.0
        elif self.profile_type == 'step':
            step_time = 0.45 * self.t_fault_start
            scale = 0.72 if t < step_time else 1.08
        elif self.profile_type == 'fluctuating':
            slow = 0.06 * np.sin(2.0 * np.pi * t / 37.0 + self.phase)
            fast = 0.025 * np.sin(2.0 * np.pi * t / 8.0 + 0.5 * self.phase)
            scale = 1.0 + slow + fast
        else:
            raise ValueError(f'Unsupported current_profile: {self.profile_type}')
        return max(0.0, self.base_current * scale)


def apply_operating_case(sim):
    door_multiplier = DOOR_STATE_MULTIPLIER.get(sim.door_state, 1.0)

    if sim.vent_state in VENT_STATE_PARAMS:
        params = VENT_STATE_PARAMS[sim.vent_state]
        sim.vent_heat_removal_scale = params['heat_removal']
        sim.vent_gas_outflow_scale = params['gas_outflow']
        sim.vent_oxygen_scale = params['oxygen']
        sim.vent_flow_multiplier = params['gas_outflow'] * (0.5 * (1.0 + door_multiplier))
        air_scale = params['heat_removal']
    else:
        vent_multiplier = VENT_STATE_MULTIPLIER.get(sim.vent_state, 1.0)
        sim.vent_heat_removal_scale = vent_multiplier
        sim.vent_gas_outflow_scale = vent_multiplier
        sim.vent_oxygen_scale = 1.0
        sim.vent_area_m2 *= vent_multiplier
        sim.vent_flow_multiplier = vent_multiplier * (0.5 * (1.0 + door_multiplier))
        air_scale = vent_multiplier

    for node in sim.nodes:
        if sim.node_type[node] in {'air_zone', 'ventilation'}:
            sim.conv_hA[node] *= air_scale
        if node == 'DOOR':
            sim.conv_hA[node] *= door_multiplier
        elif sim.node_type[node] == 'cabinet_boundary':
            sim.conv_hA[node] *= 0.5 * (1.0 + door_multiplier)


def equivalent_load_resistance(sim, curr_t):
    profile_current = sim.current_profile.value(curr_t)
    return sim.source_voltage_nominal / max(profile_current, 0.1) + sim.upstream_path_resistance, profile_current
