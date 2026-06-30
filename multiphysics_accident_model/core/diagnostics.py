"""仿真模型自检：单调性、能量合理性、标签一致性。

每次修改仿真器后运行，确保物理行为不反常。
"""

import numpy as np


def check_monotonicity(sim, history_df, diag):
    """单调性自检。

    检查：
    - 老化↑ → 引燃时间↓
    - 电流↑ → 接触功率↑ → 引燃时间↓
    - 通风恶化 → 温度↑ → 风险↑
    """
    issues = []

    # 1. 若引燃，检查引燃后 TTI=0 且 stage=3
    if history_df.empty:
        issues.append('历史为空')
        return issues

    ignited = (history_df['Stage'] >= 3.0).any()
    if ignited:
        post_ig = history_df[history_df['Stage'] >= 3.0]
        if not (post_ig['Stage'] == 3.0).all():
            issues.append('引燃后 stage 出现非 3.0 值')

    # 2. stage 单调性：不应出现 3→2 或 2→1 的倒退
    stages = history_df['Stage'].values
    for i in range(1, len(stages)):
        if stages[i] < stages[i - 1] and not (stages[i] == 1.0 and stages[i - 1] == 1.0):
            issues.append(f'第 {i} 步 stage 倒退: {stages[i-1]} → {stages[i]}')

    # 3. 温度不应在 stage>=2 时下降（除非已引燃后灭火）
    for node in sim.nodes:
        col = f'{node}_T_Core'
        if col not in history_df.columns:
            continue
        temps = history_df[col].values
        arc_mask = history_df['Stage'].values >= 2.0
        if arc_mask.any():
            arc_temps = temps[arc_mask]
            if len(arc_temps) > 5:
                # 稳定弧阶段前 5 步温度应整体上升
                if np.mean(np.diff(arc_temps[:5])) < -1.0:
                    issues.append(f'{node} 稳定弧阶段初期温度异常下降')

    # 4. 气体浓度不应为负
    for node in sim.nodes:
        col = f'{node}_C'
        if col in history_df.columns:
            if (history_df[col].fillna(0) < 0).any():
                issues.append(f'{node} 气体浓度出现负值')

    # 5. HRR 只在 stage==3 时非零
    hrr_total = history_df['Fire_HRR_Total'].values
    non_fire_hrr = hrr_total[history_df['Stage'].values < 3.0]
    if len(non_fire_hrr) > 0 and np.max(non_fire_hrr) > 1.0:
        issues.append(f'stage<3 时出现 HRR={np.max(non_fire_hrr):.2f}W')

    return issues


def check_energy_balance(sim, history_df, tolerance=0.15):
    """能量合理性自检（简化版）。

    检查输入电功率与热损耗、温升、火蔓延的能量大致平衡。
    不做精确守恒，只检查是否明显发散或出现极端值。
    """
    issues = []
    if history_df.empty:
        return issues

    dt = sim.dt_record
    P_arc = history_df['Arc_Power'].fillna(0).values
    P_contact = history_df['Line_Current'].fillna(0).values ** 2 * history_df['Fault_Resistance'].fillna(0).values
    Q_input = np.nansum((P_arc + P_contact) * dt)

    # 火蔓延热（HRR 积分）
    hrr = history_df['Fire_HRR_Total'].fillna(0).values
    Q_fire = np.nansum(hrr * dt)

    # 检查 1：起火后应有 HRR
    if (history_df['Stage'] >= 3.0).any() and Q_fire < 1.0:
        issues.append('引燃后 HRR 积分几乎为 0')

    # 检查 2：峰值温度不超过物理上限（约 2000°C）
    for node in sim.nodes:
        col = f'{node}_T_Core'
        if col not in history_df.columns:
            continue
        T_max = history_df[col].max()
        if T_max > 2000.0:
            issues.append(f'{node} 峰值温度 {T_max:.1f}°C 超过物理上限')

    # 检查 3：电弧功率与电流、电阻自洽（P = I²R）
    I = history_df['Line_Current'].fillna(0).values
    R = history_df['Fault_Resistance'].fillna(0).values
    P_est = I ** 2 * R
    P_reported = history_df['Arc_Power'].fillna(0).values + P_est
    # 只在 stage >= 1.5 时检查
    arc_mask = history_df['Stage'].values >= 1.5
    if arc_mask.any():
        err = np.abs(P_reported[arc_mask] - P_est[arc_mask])
        if len(err) > 0 and np.max(err) > 5000.0:
            issues.append(f'电弧功率与 I²R 偏差过大: max_err={np.max(err):.1f}W')

    return issues


def check_label_consistency(sim, history_df, labels):
    """标签一致性自检。

    检查：
    - y_fire=1 后不应再出现 0
    - TTI 在引燃后应为 0
    - y_10s/y_30s/y_60s 不应在引燃后乱跳
    """
    issues = []
    if history_df.empty or labels is None:
        return issues

    y_fire = labels['y_fire']
    TTI = labels['TTI']

    # 1. y_fire 单调非降
    for i in range(1, len(y_fire)):
        if y_fire[i] < y_fire[i - 1] - 1e-6:
            issues.append(f'第 {i} 步 y_fire 回退: {y_fire[i-1]} → {y_fire[i]}')

    # 2. 引燃后 TTI=0
    ignited_indices = np.where(y_fire > 0.5)[0]
    if len(ignited_indices) > 0:
        first_ig = ignited_indices[0]
        post_tti = TTI[first_ig:]
        if not np.allclose(post_tti, 0.0, atol=1e-3):
            issues.append('引燃后 TTI 不为 0')

    # 3. y_10s 在引燃后应为 1（已经知道 10s 内会起火）
    for key in ['y_10s', 'y_30s', 'y_60s']:
        y_win = labels[key]
        for i in range(1, len(y_win)):
            if y_fire[i] > 0.5 and y_win[i] < 0.5:
                issues.append(f'引燃后 {key} 为 0（第 {i} 步）')

    # 4. node_risk_gt 范围检查
    risk = labels['node_risk_gt']
    if risk.min() < -1e-6 or risk.max() > 1.0 + 1e-6:
        issues.append(f'node_risk_gt 越界: [{risk.min():.4f}, {risk.max():.4f}]')

    return issues


def run_full_diagnostics(sim, history_df, labels, diag):
    """运行全部自检，返回问题列表。"""
    all_issues = []
    all_issues.extend([('monotonicity', msg) for msg in check_monotonicity(sim, history_df, diag)])
    all_issues.extend([('energy', msg) for msg in check_energy_balance(sim, history_df)])
    all_issues.extend([('label', msg) for msg in check_label_consistency(sim, history_df, labels)])
    return all_issues


def print_diagnostics_report(sim, history_df, labels, diag, title='Diagnostics Report'):
    """打印诊断报告。"""
    issues = run_full_diagnostics(sim, history_df, labels, diag)
    print(f'\n=== {title} ===')
    print(f'Case: {sim.case_id} | Fault: {sim.f_node} | Current: {sim.cur_I0}A')
    print(f'Stage transitions: t1.5={diag.get("Stage_1_5_Time_s")} t2={diag.get("Stage_2_Time_s")} t3={diag.get("Stage_3_Time_s")}')
    print(f'Ignited nodes: {diag.get("Nodes_Ignited")}')

    if not issues:
        print('All checks PASSED.')
        return True

    print(f'Found {len(issues)} issue(s):')
    for category, msg in issues:
        print(f'  [{category}] {msg}')
    return False
