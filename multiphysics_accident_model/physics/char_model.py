def compute_char_rate(sim, node, pyro_rate):
    """碳化速率：dχ/dt = k_χ * ṁ_pyro * f(T)，其中 f(T) 在超过热解温度后激活。"""
    if pyro_rate <= 0.0 or sim.remaining_fuel[node] <= 0.0:
        return 0.0
    temp = sim.states[node][0]
    f_T = 0.0 if temp < sim.T_pyro else 1.0
    k_chi = 0.15
    return k_chi * pyro_rate * f_T
