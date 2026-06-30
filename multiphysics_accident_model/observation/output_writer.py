import numpy as np
import pandas as pd

from .label_builder import build_summary_row
from ..core.fault_profile import build_terminal_configs
from ..core.state import PIGAT_Realistic_DigitalTwin_V3


def build_model_notes():
    return pd.DataFrame([
        {'类别': '模型能力', '说明': '显式区分接触退化、微弧碳化、稳定电弧、引燃与火蔓延五个阶段。'},
        {'类别': '模型能力', '说明': '使用机械接口图中的柜体尺寸 804 x 576.5 x 1882.5 mm 构建 37 节点元器件-线束-空气区域-柜体边界混合图。'},
        {'类别': '模型能力', '说明': '导出 electrical_edges、thermal_edges、air_edges，并在 GraphEdges 中合成为 [d_mm, k_th, g_elec, a_air]。'},
        {'类别': '模型能力', '说明': '支持 FaultProfile 描述多个故障点的接触不良工况，并按 circuit_id 推断 DC/AC 回路。'},
        {'类别': '模型能力', '说明': '接触电阻拆分为金属基底、松动收缩、氧化膜和碳化通道四部分。'},
        {'类别': '模型能力', '说明': '电弧模型绑定供电模式而非节点名，DC 采用连续稳弧负阻模型，AC 采用 50 Hz 过零熄弧与重燃模型。'},
        {'类别': '模型能力', '说明': '电弧模块导出 arc_on、arc_gap_mm、arc_voltage、arc_current、arc_power、arc_energy、zero_crossings 和 reignitions。'},
        {'类别': '模型能力', '说明': '引燃条件同时考虑接触功率阈值、电弧功率阈值、气体积累、弧能和持续时间。'},
        {'类别': '模型能力', '说明': '起火后延长观察窗，并引入沿线热耦合和累计暴露判据，用于评估多节点次级点燃。'},
        {'类别': '模型局限', '说明': '接线图无法机读提取元件精确坐标，节点位置仍是基于柜体结构的工程近似。'},
        {'类别': '模型局限', '说明': '燃料质量、通风面积、换热面积和烧蚀速率尚未用实柜试验反标，只适合趋势分析。'},
        {'类别': '模型局限', '说明': '当前未显式建模电磁吹弧、滴落引燃和局部熔珠飞溅，极端工况下仍可能低估瞬态危险。'},
        {'类别': '模型局限', '说明': '输出为半物理数字孪生，不应直接替代型式试验或消防认证测试。'},
    ])


def run_sweep(output_filename='fangzhen_digital_twin_v5.xlsx', currents=None, fault_terminals=None, case_ids=None, service_years=None, vent_state=None):
    summary = []
    diagnostics_rows = []

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        if case_ids is not None:
            terminal_configs = build_terminal_configs()
            for case_id in case_ids:
                cfg = terminal_configs[case_id]
                current = cfg['current_level']
                case_service_years = cfg['service_years'] if service_years is None else service_years
                seed = cfg['random_seed']
                sim = PIGAT_Realistic_DigitalTwin_V3(
                    init_current=current,
                    case_id=case_id,
                    service_days=30,
                    service_years=case_service_years,
                    seed=seed,
                    vent_state=vent_state,
                )
                df, t1_5, t2, t3, q_at_ig, diag = sim.run()
                terminal = sim.fault_site
                sheet_name = f'{diag["Case_ID"]}_{int(current)}A'
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                summary.append(build_summary_row(terminal, current, t1_5, t2, t3, q_at_ig, diag))
                diagnostics_rows.append(diag)
                status = '未起火' if t3 is None else '起火'
                t1_5_str = 'None' if t1_5 is None else f'{t1_5:.3f}'
                t2_str = 'None' if t2 is None else f'{t2:.3f}'
                t3_str = 'None' if t3 is None else f'{t3:.3f}'
                print(
                    f' [√] {terminal:6s} | {current:4.1f}A | 微弧: {t1_5_str:>8s} | 稳定弧: {t2_str:>8s} '
                    f'| 起火: {t3_str:>8s} | 峰值弧功: {diag["Peak_Arc_Power_W"]:>7.1f} W | {status}'
                )
        else:
            currents = list(np.arange(0.0, 40.1, 2.0)) if currents is None else currents
            fault_terminals = ['KM1_DC', 'KM1_AC', 'X1', 'X2'] if fault_terminals is None else fault_terminals
            for terminal in fault_terminals:
                for current in currents:
                    sim = PIGAT_Realistic_DigitalTwin_V3(
                        init_current=current,
                        fault_terminal=terminal,
                        case_id=None,
                        service_days=30,
                        service_years=service_years,
                        seed=int(current * 1000 + (0 if terminal == 'KM1_DC' else 37)),
                        vent_state=vent_state,
                    )
                    df, t1_5, t2, t3, q_at_ig, diag = sim.run()
                    terminal = sim.fault_site
                    sheet_name = f'{diag["Case_ID"]}_{int(current)}A'
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    summary.append(build_summary_row(terminal, current, t1_5, t2, t3, q_at_ig, diag))
                    diagnostics_rows.append(diag)
                    status = '未起火' if t3 is None else '起火'
                    t1_5_str = 'None' if t1_5 is None else f'{t1_5:.3f}'
                    t2_str = 'None' if t2 is None else f'{t2:.3f}'
                    t3_str = 'None' if t3 is None else f'{t3:.3f}'
                    print(
                        f' [√] {terminal:6s} | {current:4.1f}A | 微弧: {t1_5_str:>8s} | 稳定弧: {t2_str:>8s} '
                        f'| 起火: {t3_str:>8s} | 峰值弧功: {diag["Peak_Arc_Power_W"]:>7.1f} W | {status}'
                    )

        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(diagnostics_rows).to_excel(writer, sheet_name='Diagnostics', index=False)
        graph_meta = PIGAT_Realistic_DigitalTwin_V3(init_current=10.0, fault_terminal='KM1_DC', service_days=30, service_years=service_years, seed=0)
        graph_meta.get_case_config_df().to_excel(writer, sheet_name='CaseConfig', index=False)
        graph_meta.get_graph_nodes_df().to_excel(writer, sheet_name='GraphNodes', index=False)
        graph_meta.get_graph_edges_df().to_excel(writer, sheet_name='GraphEdges', index=False)
        graph_meta.get_electrical_edges_df().to_excel(writer, sheet_name='ElectricalEdges', index=False)
        graph_meta.get_thermal_edges_df().to_excel(writer, sheet_name='ThermalEdges', index=False)
        graph_meta.get_air_edges_df().to_excel(writer, sheet_name='AirEdges', index=False)
        graph_meta.get_aging_profiles_df().to_excel(writer, sheet_name='AgingProfiles', index=False)
        build_model_notes().to_excel(writer, sheet_name='ModelNotes', index=False)

    return output_filename
