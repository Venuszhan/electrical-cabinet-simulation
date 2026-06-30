from ..physics.pyrolysis_model import compute_gas_source_rate


def update_gas_smoke_step(sim, next_t, pyro_rates, gas_source_rates, char_rates):
    """气体浓度 + 碳化指数 + 传感器温度联合更新。

    质量路径：
        remaining_fuel --pyrolysis--> volatile_pool + char + loss
        volatile_pool --release--> C_raw
        C_raw --diffusion/leak/combustion_sink--> 更新

    这样既保留燃料有限性，又避免热解质量直接从系统中消失。
    """
    C_snapshot = {node: sim.states[node][1] for node in sim.nodes}

    for node in sim.nodes:
        # 1. core temperature update
        sim.states[node][0] = next_t[node]
        sim.peak_core_temp = max(sim.peak_core_temp, sim.states[node][0])

        node_type = getattr(sim, 'node_type', {}).get(node, '')
        if node_type == 'air_zone':
            sim.peak_T_air_zone = max(sim.peak_T_air_zone, sim.states[node][0])
        elif node != sim.f_node:
            sim.peak_T_solid_non_air = max(sim.peak_T_solid_non_air, sim.states[node][0])

        # 2. sensor temperature lag
        d_t_sensor = (sim.states[node][0] - sim.sensor_T[node]) / sim.tau_sensor
        sim.sensor_T[node] += d_t_sensor * sim.dt_record

        # 3. solid fuel -> volatile_pool / char / loss
        raw_dm_pyro = max(0.0, pyro_rates[node]) * sim.dt_record
        dm_pyro = min(raw_dm_pyro, max(sim.remaining_fuel[node], 0.0))

        if dm_pyro > 0.0:
            sim.remaining_fuel[node] -= dm_pyro
            sim.remaining_fuel[node] = max(0.0, sim.remaining_fuel[node])

            sim.pyrolyzed_mass[node] += dm_pyro
            sim.pyrolyzed_mass[node] = min(
                sim.pyrolyzed_mass[node],
                sim.fuel_mass_total[node],
            )

            volatile_yield = getattr(sim, 'pyro_volatile_yield', 0.62)
            char_yield = getattr(sim, 'pyro_char_yield', 0.22)

            sim.volatile_pool[node] += volatile_yield * dm_pyro
            volatile_cap = (
                getattr(sim, 'volatile_pool_cap_factor', 1.20)
                * max(sim.fuel_mass_total[node], 1e-6)
            )
            sim.volatile_pool[node] = min(sim.volatile_pool[node], volatile_cap)

            sim.volatile_generated_mass[node] += volatile_yield * dm_pyro

            sim.volatile_pool_peak[node] = max(
                sim.volatile_pool_peak[node],
                sim.volatile_pool[node],
            )

            dm_char = char_yield * dm_pyro
            sim.char_fuel_mass[node] += dm_char
            char_from_mass = dm_char / max(sim.fuel_mass_total[node], 1e-6)
        else:
            char_from_mass = 0.0

        # 4. char index accumulation
        sim.char_index[node] += char_rates[node] * sim.dt_record + 0.35 * char_from_mass
        sim.char_index[node] = min(1.0, sim.char_index[node])

        # 5. volatile_pool -> gas release
        T_c = sim.states[node][0]
        temp_drive = max(0.0, T_c - sim.T_pyro) / 180.0

        release_base = getattr(sim, 'volatile_release_base', 0.28)
        release_temp_scale = getattr(sim, 'volatile_release_temp_scale', 0.85)

        release_coeff = release_base * (
            0.20 + release_temp_scale * min(2.0, temp_drive)
        )

        dm_release = min(
            sim.volatile_pool[node],
            release_coeff * sim.volatile_pool[node] * sim.dt_record,
        )
        sim.volatile_pool[node] -= dm_release
        sim.volatile_pool[node] = max(0.0, sim.volatile_pool[node])

        sim.volatile_release_exposure[node] += dm_release

        release_rate = dm_release / sim.dt_record if sim.dt_record > 0 else 0.0
        q_gas = compute_gas_source_rate(sim, node, release_rate)

        # 少量即时释放项，避免所有气体都滞留在 volatile_pool
        if dm_pyro > 0.0:
            pyro_rate_eff = dm_pyro / sim.dt_record
            q_gas += 0.20 * compute_gas_source_rate(sim, node, pyro_rate_eff)

        # 6. diffusion
        diffusion = 0.0
        air_neighbors = getattr(sim, 'air_neighbors', {}).get(node, [])
        if air_neighbors:
            for nbr, a_air, d_gas, _flow_dir in air_neighbors:
                dist_m = sim.get_dist(node, nbr) / 1000.0
                D_eff = d_gas * a_air / max(dist_m, 0.08)
                diffusion += D_eff * (C_snapshot[nbr] - C_snapshot[node])
        else:
            for nbr in sim.nodes:
                if nbr == node:
                    continue
                dist_m = sim.get_dist(node, nbr) / 1000.0
                D_eff = sim.gas_diffusion / max(dist_m, 0.08)
                diffusion += D_eff * (C_snapshot[nbr] - C_snapshot[node])

        # 7. leak / outflow
        leak = (
            sim.k_leak
            * getattr(sim, 'vent_gas_outflow_scale', 1.0)
            * sim.states[node][1]
        )

        # 8. combustion sink
        # volatile_pool 的燃烧消耗由 update_combustion 统一处理，
        # 这里只保留 C_raw 的燃烧沉降项。
        combustion_sink = 0.0
        if sim.node_ignited[node]:
            combustion_sink = 0.25 * sim.node_hrr[node] / 1000.0

        # 9. C_raw update
        dC = q_gas + diffusion - leak - combustion_sink
        sim.states[node][1] += dC * sim.dt_record
        sim.states[node][1] = max(0.0, sim.states[node][1])

        # 10. gas exposure accumulation
        c_drive = max(0.0, sim.states[node][1] - getattr(sim, 'gas_exposure_base', 0.2))
        sim.gas_exposure[node] += c_drive * sim.dt_record
