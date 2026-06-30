import numpy as np


def generate_ar_noise(sim, key, sub_key=None, std=0.1):
    if sub_key is not None:
        prev_noise = sim.noise_states[key][sub_key]
        new_noise = sim.ar_phi * prev_noise + sim.rng.normal(0, std * (1 - sim.ar_phi ** 2) ** 0.5)
        sim.noise_states[key][sub_key] = new_noise
        return new_noise

    prev_noise = sim.noise_states[key]
    new_noise = sim.ar_phi * prev_noise + sim.rng.normal(0, std * (1 - sim.ar_phi ** 2) ** 0.5)
    sim.noise_states[key] = new_noise
    return new_noise


def apply_sensor_dropout(value, dropout_prob=0.015, rng=None):
    """随机丢包：以 dropout_prob 概率返回 NaN。"""
    if rng is not None:
        if rng.random() < dropout_prob:
            return np.nan
    elif np.random.random() < dropout_prob:
        return np.nan
    return value


def apply_gas_sensor_noise(C_true, rng=None, base_std=0.6, low_conc_boost=3.0, saturation_level=120.0):
    """气体/烟雾传感器噪声模型：
    - 低浓度时噪声被放大（基线漂移影响更大）
    - 高浓度时逐渐趋于饱和
    """
    if C_true <= 0.0:
        std = base_std * low_conc_boost
    else:
        std = base_std * (1.0 + (low_conc_boost - 1.0) / (1.0 + 0.08 * C_true))
    if rng is not None:
        noise = rng.normal(0, std)
    else:
        noise = np.random.normal(0, std)
    meas = C_true + noise
    return float(min(max(meas, 0.0), saturation_level))


def apply_temperature_sensor_lag(T_core, T_sensor_prev, tau_sensor, dt):
    """温度传感器一阶滞后。"""
    dT = (T_core - T_sensor_prev) / tau_sensor
    return T_sensor_prev + dT * dt


def apply_sensor_drift(drift_state, drift_rate=0.0005, rng=None):
    """传感器漂移：随机游走偏置。"""
    step = rng.normal(0, drift_rate) if rng is not None else np.random.normal(0, drift_rate)
    return drift_state + step


def apply_sampling_jitter(dt_nominal, jitter_frac=0.05, rng=None):
    """采样间隔抖动：在标称间隔上叠加 ±jitter_frac 的相对随机偏移。"""
    if rng is not None:
        jitter = rng.uniform(-jitter_frac, jitter_frac)
    else:
        jitter = np.random.uniform(-jitter_frac, jitter_frac)
    return dt_nominal * (1.0 + jitter)
