"""PIGAT 监督标签生成器。

为每个时间步输出丰富的标签，用于 PIGAT 训练：
- y_fire / y_stage / TTI：全局时间序列标签
- node_risk_gt：节点级风险真值
- y_10s / y_30s / y_60s：未来窗口起火预测标签
- node_feature_names：节点特征维度名列表
"""

import numpy as np


NODE_FEATURE_NAMES = [
    'V',      # 0: 电压
    'I',      # 1: 电流
    'P',      # 2: 电功率
    'R',      # 3: 电阻
    'T',      # 4: 温度
    'C',      # 5: 气体浓度
    'HRR',    # 6: 热释放率
    'dT',     # 7: 温升速率
    'dI',     # 8: 电流变化率
    'dC',     # 9: 浓度变化率
    'tau',    # 10: 服役年限
    'eta_con',# 11: 接触老化因子
    'eta_ox', # 12: 氧化老化因子
    'eta_ins',# 13: 绝缘老化因子
    'eta_fuel',# 14: 燃料老化因子
    'Ea',     # 15: 活化能 (kJ/mol)
]

N_NODE_FEATURES = len(NODE_FEATURE_NAMES)

EDGE_FEATURE_NAMES = [
    'k_th',            # 0: 热导
    'g_elec',          # 1: 电导
    'a_air',           # 2: 气流权重
    'inv_dist_mm',     # 3: 距离倒数
]

N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)


def _compute_derivatives(series, dt):
    """计算一阶差分（导数），首点用前向差分。"""
    if len(series) < 2:
        return np.zeros_like(series)
    d = np.diff(series) / dt
    return np.concatenate([[d[0]], d])


def _safe_get(row, key, default):
    v = row.get(key, default)
    if v != v:  # NaN check for floats
        return default
    return v


def _node_risk_score(row, node, f_node, ambient_temp):
    """单节点风险分：综合温度、浓度、HRR、与故障端距离、阶段。"""
    T = _safe_get(row, f'{node}_T_Core', ambient_temp)
    C = _safe_get(row, f'{node}_C', 0.0)
    HRR = _safe_get(row, f'{node}_HRR', 0.0)
    stage = _safe_get(row, 'Stage', 1.0)

    risk = 0.0
    # 温度风险
    risk += np.clip((T - ambient_temp) / 300.0, 0.0, 1.0) * 0.25
    # 浓度风险
    risk += np.clip(C / 80.0, 0.0, 1.0) * 0.20
    # HRR 风险
    risk += np.clip(HRR / 5000.0, 0.0, 1.0) * 0.30
    # 阶段风险
    if stage >= 3.0:
        risk += 0.25
    elif stage >= 2.0:
        risk += 0.15
    elif stage >= 1.5:
        risk += 0.08

    # 故障端额外加权
    if node == f_node:
        risk = min(1.0, risk * 1.35)

    return float(np.clip(risk, 0.0, 1.0))


def build_pigate_labels(df, sim, dt_record=0.02):
    """从仿真历史 DataFrame 生成 PIGAT 监督标签。

    返回 dict：
    {
        'y_fire': np.array[T],              # 0/1 是否已起火
        'y_stage': np.array[T],             # 1.0/1.5/2.0/3.0
        'TTI': np.array[T],                 # 距起火剩余时间 (s)，已起火为 0
        'node_risk_gt': np.array[T, N],     # 节点级风险 [0,1]
        'y_10s': np.array[T],               # 未来 10s 内是否起火
        'y_30s': np.array[T],               # 未来 30s 内是否起火
        'y_60s': np.array[T],               # 未来 60s 内是否起火
        'ignition_time': float or None,     # 起火时间
        'f_node_idx': int,                  # 故障节点索引
    }
    """
    if df.empty:
        return None

    T_steps = len(df)
    N_nodes = len(sim.nodes)
    node_id = {node: idx for idx, node in enumerate(sim.nodes)}
    f_node_idx = node_id[sim.f_node]

    times = df['Time'].values
    stages = df['Stage'].values

    # --- 全局标签 ---
    y_fire = (stages >= 3.0).astype(np.float32)
    y_stage = stages.astype(np.float32)

    # TTI: 找到起火时刻，向后填充 0，向前线性插值
    ignition_mask = y_fire > 0.5
    if ignition_mask.any():
        t_ign = times[ignition_mask][0]
        TTI = np.where(times >= t_ign, 0.0, t_ign - times).astype(np.float32)
    else:
        t_ign = None
        TTI = np.full(T_steps, -1.0, dtype=np.float32)  # -1 表示未起火

    # --- 窗口预测标签 ---
    def _window_label(future_seconds):
        win_steps = int(future_seconds / dt_record)
        labels = np.zeros(T_steps, dtype=np.float32)
        for i in range(T_steps):
            j = min(i + win_steps, T_steps - 1)
            labels[i] = float((stages[i:j+1] >= 3.0).any())
        return labels

    y_10s = _window_label(10.0)
    y_30s = _window_label(30.0)
    y_60s = _window_label(60.0)

    # --- 节点级风险真值 ---
    node_risk_gt = np.zeros((T_steps, N_nodes), dtype=np.float32)
    for t_idx, (_, row) in enumerate(df.iterrows()):
        for node, idx in node_id.items():
            node_risk_gt[t_idx, idx] = _node_risk_score(row, node, sim.f_node, sim.ambient_temp)

    return {
        'y_fire': y_fire,
        'y_stage': y_stage,
        'TTI': TTI,
        'node_risk_gt': node_risk_gt,
        'y_10s': y_10s,
        'y_30s': y_30s,
        'y_60s': y_60s,
        'ignition_time': t_ign,
        'f_node_idx': f_node_idx,
        'node_id': node_id,
        'nodes': sim.nodes,
    }


def build_node_feature_matrix(df, sim, dt_record=0.02):
    """构建节点特征矩阵 x: [T, N, F_node]。

    特征顺序与 NODE_FEATURE_NAMES 一致。
    """
    if df.empty:
        return None

    T_steps = len(df)
    N_nodes = len(sim.nodes)
    node_id = {node: idx for idx, node in enumerate(sim.nodes)}

    x = np.zeros((T_steps, N_nodes, N_NODE_FEATURES), dtype=np.float32)

    for node, idx in node_id.items():
        v_col = f'{node}_V'
        t_col = f'{node}_T_Core'
        c_col = f'{node}_C'
        hrr_col = f'{node}_HRR'

        V = df[v_col].fillna(0.0).values.astype(np.float32)
        # 电流：若节点在串联路径上取全局电流，否则 0
        I = np.zeros(T_steps, dtype=np.float32)
        if node in sim.series_path_nodes:
            I = df['Line_Current'].fillna(0.0).values.astype(np.float32)
        P = V * I
        R = df['Fault_Resistance'].fillna(0.0).values.astype(np.float32) if node == sim.f_node else np.full(T_steps, sim.node_resistance.get(node, 0.0), dtype=np.float32)
        T = df[t_col].fillna(sim.ambient_temp).values.astype(np.float32)
        C = df[c_col].fillna(0.0).values.astype(np.float32)
        HRR = df[hrr_col].fillna(0.0).values.astype(np.float32)

        dT = _compute_derivatives(T, dt_record)
        dI = _compute_derivatives(I, dt_record)
        dC = _compute_derivatives(C, dt_record)

        aging = sim.aging_profiles[node]
        tau = sim.service_years
        eta_con = aging.eta_contact
        eta_ox = aging.eta_oxide
        eta_ins = aging.eta_insulation
        eta_fuel = aging.eta_fuel
        Ea = aging.Ea / 1000.0

        x[:, idx, 0] = V
        x[:, idx, 1] = I
        x[:, idx, 2] = P
        x[:, idx, 3] = R
        x[:, idx, 4] = T
        x[:, idx, 5] = C
        x[:, idx, 6] = HRR
        x[:, idx, 7] = dT
        x[:, idx, 8] = dI
        x[:, idx, 9] = dC
        x[:, idx, 10] = tau
        x[:, idx, 11] = eta_con
        x[:, idx, 12] = eta_ox
        x[:, idx, 13] = eta_ins
        x[:, idx, 14] = eta_fuel
        x[:, idx, 15] = Ea

    return x


def build_graph_static_tensors(sim):
    """构建静态图张量：edge_feat[N,N,4], adj[N,N], k_ij[N,N], D_ij[N,N]。

    这些张量不随时间变化，只与拓扑结构有关。
    """
    N = len(sim.nodes)
    node_id = {node: idx for idx, node in enumerate(sim.nodes)}

    edge_feat = np.zeros((N, N, N_EDGE_FEATURES), dtype=np.float32)
    adj = np.zeros((N, N), dtype=np.float32)
    k_ij = np.zeros((N, N), dtype=np.float32)
    D_ij = np.zeros((N, N), dtype=np.float32)

    for edge in sim.graph_edges:
        src, dst = edge['src'], edge['dst']
        if src not in node_id or dst not in node_id:
            continue
        i, j = node_id[src], node_id[dst]

        d_mm = float(edge.get('distance_mm', sim.get_dist(src, dst)))
        k_th = float(edge.get('k_th', 0.0))
        g_elec = float(edge.get('g_elec', 0.0))
        a_air = float(edge.get('a_air', 0.0))
        d_gas = edge.get('D_gas')

        adj[i, j] = 1.0
        adj[j, i] = 1.0
        edge_feat[i, j, 0] = k_th
        edge_feat[j, i, 0] = k_th
        edge_feat[i, j, 1] = g_elec
        edge_feat[j, i, 1] = g_elec
        edge_feat[i, j, 2] = a_air
        edge_feat[j, i, 2] = a_air
        edge_feat[i, j, 3] = 1.0 / max(d_mm, 1.0)
        edge_feat[j, i, 3] = 1.0 / max(d_mm, 1.0)

        if k_th > 0.0:
            k_ij[i, j] = k_th
            k_ij[j, i] = k_th

        if d_gas is not None and a_air > 0.0:
            d_gas = float(d_gas)
            D_eff = d_gas * a_air / max(d_mm / 1000.0, 0.08)
            D_ij[i, j] = D_eff
            D_ij[j, i] = D_eff

    # 自环
    for i in range(N):
        adj[i, i] = 1.0
        edge_feat[i, i, 3] = 1.0

    return {
        'edge_feat': edge_feat,
        'adj': adj,
        'k_ij': k_ij,
        'D_ij': D_ij,
        'node_id': node_id,
        'nodes': sim.nodes,
    }
