"""PIGAT 数据适配器：将仿真历史 DataFrame + 仿真对象转换为 PIGAT 输入张量。

输出格式对齐 PIGAT 模型期望：
- x_window:  [W, N, F_node]   节点时序特征
- edge_feat: [N, N, 4]        边静态特征
- adj:       [N, N]           邻接矩阵
- k_ij:      [N, N]           热导矩阵
- D_ij:      [N, N]           扩散矩阵
- y_fire:    [T]              全局起火标签
- y_stage:   [T]              阶段标签
- TTI:       [T]              距起火时间
- node_risk_gt: [T, N]        节点级风险真值
"""

import numpy as np

from .pigate_labels import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_graph_static_tensors,
    build_node_feature_matrix,
    build_pigate_labels,
)


def convert_case_to_pigate(sim, history_df, dt_record=0.02):
    """将一次仿真结果转换为 PIGAT 数据包。

    返回 dict，包含所有 PIGAT 需要的张量和元数据。
    """
    if history_df.empty:
        return None

    labels = build_pigate_labels(history_df, sim, dt_record)
    x = build_node_feature_matrix(history_df, sim, dt_record)
    graph_tensors = build_graph_static_tensors(sim)

    return {
        'x_window': x,                       # [T, N, F_node]
        'edge_feat': graph_tensors['edge_feat'],  # [N, N, 4]
        'adj': graph_tensors['adj'],         # [N, N]
        'k_ij': graph_tensors['k_ij'],       # [N, N]
        'D_ij': graph_tensors['D_ij'],       # [N, N]
        'y_fire': labels['y_fire'],          # [T]
        'y_stage': labels['y_stage'],        # [T]
        'TTI': labels['TTI'],                # [T]
        'node_risk_gt': labels['node_risk_gt'],  # [T, N]
        'y_10s': labels['y_10s'],            # [T]
        'y_30s': labels['y_30s'],            # [T]
        'y_60s': labels['y_60s'],            # [T]
        'f_node_idx': labels['f_node_idx'],  # int
        'ignition_time': labels['ignition_time'],  # float or None
        'node_id': labels['node_id'],        # dict
        'nodes': labels['nodes'],            # list
        'node_feature_names': NODE_FEATURE_NAMES,
        'edge_feature_names': EDGE_FEATURE_NAMES,
        'case_id': sim.case_id,
        'fault_terminal': sim.fault_site,
        'fault_site': sim.fault_site,
        'representative_fault_node': sim.f_node,
        'current_level': sim.cur_I0,
        'service_years': sim.service_years,
        'vent_state': sim.vent_state,
        'door_state': sim.door_state,
        'ambient_temp': sim.ambient_temp,
    }


def save_pigate_data(data, filepath):
    """将 PIGAT 数据包保存为 .npz 文件。"""
    if data is None:
        return
    np.savez_compressed(
        filepath,
        x_window=data['x_window'],
        edge_feat=data['edge_feat'],
        adj=data['adj'],
        k_ij=data['k_ij'],
        D_ij=data['D_ij'],
        y_fire=data['y_fire'],
        y_stage=data['y_stage'],
        TTI=data['TTI'],
        node_risk_gt=data['node_risk_gt'],
        y_10s=data['y_10s'],
        y_30s=data['y_30s'],
        y_60s=data['y_60s'],
        f_node_idx=data['f_node_idx'],
        ignition_time=data['ignition_time'] if data['ignition_time'] is not None else -1.0,
        case_id=str(data['case_id']),
        fault_terminal=str(data['fault_terminal']),
        fault_site=str(data['fault_site']),
        representative_fault_node=str(data['representative_fault_node']),
        nodes=np.array(data['nodes'], dtype=object),
        node_feature_names=np.array(data['node_feature_names'], dtype=object),
        edge_feature_names=np.array(data['edge_feature_names'], dtype=object),
        current_level=float(data['current_level']),
        service_years=float(data['service_years']),
        vent_state=str(data['vent_state']),
        door_state=str(data['door_state']),
        ambient_temp=float(data['ambient_temp']),
    )


def load_pigate_data(filepath):
    """从 .npz 文件加载 PIGAT 数据包。"""
    raw = np.load(filepath, allow_pickle=True)
    return {
        'x_window': raw['x_window'],
        'edge_feat': raw['edge_feat'],
        'adj': raw['adj'],
        'k_ij': raw['k_ij'],
        'D_ij': raw['D_ij'],
        'y_fire': raw['y_fire'],
        'y_stage': raw['y_stage'],
        'TTI': raw['TTI'],
        'node_risk_gt': raw['node_risk_gt'],
        'y_10s': raw['y_10s'],
        'y_30s': raw['y_30s'],
        'y_60s': raw['y_60s'],
        'f_node_idx': int(raw['f_node_idx']),
        'ignition_time': float(raw['ignition_time']) if raw['ignition_time'] >= 0 else None,
        'case_id': str(raw['case_id']),
        'fault_terminal': str(raw['fault_terminal']),
        'fault_site': str(raw['fault_site']),
        'representative_fault_node': str(raw['representative_fault_node']),
        'nodes': raw['nodes'].tolist(),
        'node_feature_names': raw['node_feature_names'].tolist(),
        'edge_feature_names': raw['edge_feature_names'].tolist(),
        'current_level': float(raw['current_level']),
        'service_years': float(raw['service_years']),
        'vent_state': str(raw['vent_state']),
        'door_state': str(raw['door_state']),
        'ambient_temp': float(raw['ambient_temp']),
    }
