if __package__ in {None, ''}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multiphysics_accident_model.observation.output_writer import run_sweep
else:
    from .observation.output_writer import run_sweep


def parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(',') if item.strip()]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the cabinet fire digital-twin accident cases.')
    parser.add_argument('--output', default='fangzhen_digital_twin_v5.xlsx')
    parser.add_argument('--currents', type=parse_csv_floats, default=None, help='Comma-separated current list, e.g. 0,2,4.')
    parser.add_argument('--fault-terminals', type=parse_csv_strings, default=None, help='Comma-separated fault sites, e.g. KM1_DC,KM1_AC,X1,X2.')
    parser.add_argument('--case-ids', type=parse_csv_strings, default=None, help='Comma-separated case IDs from config/case_table.csv.')
    parser.add_argument('--service-years', type=float, default=None, help='Continuous service age in years, clipped to 0..60.')
    parser.add_argument('--vent', default=None, help='Override vent_state for every case (e.g. normal, blocked_30, blocked_60, fan_failed).')
    args = parser.parse_args()

    print('>>> 启动电气柜接触不良-电弧-火蔓延数字孪生仿真台 (V5.0)...')
    print('>>> 引擎特性: 柜体尺寸约束 | 接触退化 | 微弧碳化桥 | AC/DC 分叉弧模型 | 非固定热释放 | 多节点蔓延诊断')
    run_sweep(
        output_filename=args.output,
        currents=args.currents,
        fault_terminals=args.fault_terminals,
        case_ids=args.case_ids,
        service_years=args.service_years,
        vent_state=args.vent,
    )
    print(f'\n>>> 仿真完成，结果已写入 {args.output}。')
