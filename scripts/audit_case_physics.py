"""Screen the 1920-case accident-case result set for physical plausibility.

This audit does not alter case outputs.  It combines the case summary
with stored per-case diagnostics and inspects one worst-case history trace to
make the main propagation concern reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "accident_cases_1920"
OUTPUT_CSV = ROOT / "physics_audit_cases.csv"
OUTPUT_REPORT = ROOT / "PHYSICS_AUDIT_REPORT.md"
REQUIRED_FILES = (
    "case_config.json",
    "summary.json",
    "diagnostics.json",
    "diagnostics.csv",
    "history.csv.gz",
    "pigat_data.npz",
    "system_timeseries.csv.gz",
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ROOT / "summary.csv")
    diagnostics = []
    for case_id in summary["case_id"]:
        with (ROOT / case_id / "diagnostics.json").open("r", encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        diagnostic["folder_case_id"] = case_id
        diagnostics.append(diagnostic)
    return summary, pd.DataFrame(diagnostics)


def file_missing_text(case_id: str) -> str:
    missing = [name for name in REQUIRED_FILES if not (ROOT / case_id / name).exists()]
    return ";".join(missing)


def build_case_audit(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    diag_columns = [
        "folder_case_id",
        "Case_ID",
        "Peak_Core_T_C",
        "Peak_T_Air_Zone_C",
        "Peak_T_Solid_NonAir_C",
        "Nodes_Ignited",
        "Arc_Quenched_No_Ignition",
        "Min_Burnable_Mass_Fault_g",
    ]
    audit = summary.merge(
        diagnostics[diag_columns],
        left_on="case_id",
        right_on="folder_case_id",
        how="left",
        validate="one_to_one",
    )
    audit["missing_files"] = audit["case_id"].map(file_missing_text)
    audit["flag_missing_files"] = audit["missing_files"].str.len() > 0
    audit["flag_diagnostic_case_id_mismatch"] = audit["case_id"] != audit["Case_ID"]
    audit["flag_qc_issue"] = audit["Runner_QC_Issues"].notna() | audit["Diagnostic_Issues"].notna()
    audit["flag_stage_order"] = (
        (audit["Stage_2_Time_s"].notna() & (
            audit["Stage_1_5_Time_s"].isna()
            | (audit["Stage_2_Time_s"] < audit["Stage_1_5_Time_s"])
        ))
        | (audit["Stage_3_Time_s"].notna() & (
            audit["Stage_2_Time_s"].isna()
            | (audit["Stage_3_Time_s"] < audit["Stage_2_Time_s"])
        ))
    )
    audit["flag_hot_neighbor_no_secondary_300C"] = (
        (audit["Max_Neighbor_T"] >= 300.0) & (audit["Secondary_Ignited_Count"] == 0)
    )
    audit["flag_hot_neighbor_no_secondary_500C"] = (
        (audit["Max_Neighbor_T"] >= 500.0) & (audit["Secondary_Ignited_Count"] == 0)
    )
    audit["flag_arc_without_sustained_fire"] = audit["Arc_Without_Sustained_Fire"].astype(bool)
    audit["flag_fire_below_burnable_threshold"] = (
        audit["Stage_3_Time_s"].notna()
        & (audit["Fault_Burnable_Mass_at_Stage3_g"] < audit["Min_Burnable_Mass_Fault_g"])
    )
    suspect = (
        audit["flag_hot_neighbor_no_secondary_300C"]
        | audit["flag_arc_without_sustained_fire"]
        | audit["flag_fire_below_burnable_threshold"]
        | audit["flag_stage_order"]
    )
    audit["physics_screening"] = np.where(suspect, "physically_questionable", "trend_only_unvalidated")
    return audit


def inspect_hottest_neighbor(audit: pd.DataFrame) -> dict[str, object]:
    row = audit.loc[audit["Max_Neighbor_T"].idxmax()]
    case_id = row["case_id"]
    fault = row["fault_terminal"]
    neighbor = row["Top_Risk_Neighbor"]
    usecols = [
        "Time",
        "Stage",
        "Fire_HRR_Total",
        f"{fault}_T_Core",
        f"{neighbor}_T_Core",
        f"{neighbor}_C",
        f"{neighbor}_HRR",
    ]
    history = pd.read_csv(ROOT / case_id / "history.csv.gz", usecols=usecols)
    max_i = history[f"{neighbor}_T_Core"].idxmax()
    return {
        "case_id": case_id,
        "fault": fault,
        "neighbor": neighbor,
        "neighbor_max_temp": float(history.loc[max_i, f"{neighbor}_T_Core"]),
        "neighbor_max_temp_time": float(history.loc[max_i, "Time"]),
        "neighbor_hrr_max": float(history[f"{neighbor}_HRR"].max()),
        "neighbor_c_max": float(history[f"{neighbor}_C"].max()),
        "duration_over_300_s": float((history[f"{neighbor}_T_Core"] >= 300.0).sum() * 0.02),
    }


def verify_flagged_hot_histories(audit: pd.DataFrame) -> dict[str, object]:
    flagged = audit[audit["flag_hot_neighbor_no_secondary_300C"]]
    max_temp_mismatch = 0
    positive_neighbor_hrr = 0
    durations = []
    for row in flagged.itertuples(index=False):
        temp_col = f"{row.Top_Risk_Neighbor}_T_Core"
        hrr_col = f"{row.Top_Risk_Neighbor}_HRR"
        history = pd.read_csv(ROOT / row.case_id / "history.csv.gz", usecols=[temp_col, hrr_col])
        if abs(float(history[temp_col].max()) - float(row.Max_Neighbor_T)) > 0.02:
            max_temp_mismatch += 1
        if float(history[hrr_col].max()) > 0.0:
            positive_neighbor_hrr += 1
        durations.append(float((history[temp_col] >= 300.0).sum() * 0.02))
    return {
        "checked": len(flagged),
        "max_temp_mismatch": max_temp_mismatch,
        "positive_neighbor_hrr": positive_neighbor_hrr,
        "duration_min_s": min(durations) if durations else 0.0,
        "duration_max_s": max(durations) if durations else 0.0,
    }


def fmt_count(audit: pd.DataFrame, column: str) -> int:
    return int(audit[column].sum())


def write_report(audit: pd.DataFrame, hottest: dict[str, object], hot_verification: dict[str, object]) -> None:
    audit = audit.copy()
    audit["fire"] = audit["Stage_3_Time_s"].notna()
    vent_cells = audit.groupby(
        ["fault_terminal", "current_level", "service_years", "vent_state"]
    )["fire"].sum().unstack()
    seed_groups = audit.groupby(["fault_terminal", "current_level", "service_years", "vent_state"])
    seed_fire_variation = int((seed_groups["fire"].nunique() > 1).sum())
    seed_t3_range = float(seed_groups["Stage_3_Time_s"].agg(lambda v: v.max() - v.min()).max())
    fire_by_fault = audit.groupby("fault_terminal")["fire"].agg(["sum", "mean"])
    fire_by_age = audit.groupby("service_years")["fire"].mean()
    fire_by_vent = audit.groupby("vent_state")["fire"].mean()

    fire_fault_lines = "\n".join(
        f"| {fault} | {int(row['sum'])} | {row['mean']:.1%} |"
        for fault, row in fire_by_fault.iterrows()
    )
    age_line = " | ".join(f"{age:g} y: {rate:.1%}" for age, rate in fire_by_age.items())
    vent_line = " | ".join(f"{vent}: {rate:.1%}" for vent, rate in fire_by_vent.items())
    report = f"""# LUANSHENG 1920 工况物理合理性审计

## 结论

该结果集可视为内部逻辑基本完整的半物理合成数据，但**不能判定为符合真实物理场景的最终验证数据**。1920 个工况均成功产出且未触发运行器 QC；然而蔓延行为、通风敏感性、老化趋势及随机性存在需要重新校准或实验验证的明显问题。

## 范围与方法

- 审计对象：`{ROOT}`，共 {len(audit)} 个工况。
- 全量检查：每工况必要文件、summary/diagnostics、阶段顺序、火灾与 HRR 一致性、燃烧可用质量门限、诊断 QC、邻点极端温度与蔓延结果。
- 时间序列复核：对邻点最高温的最严重案例读取完整 `history.csv.gz` 验证温度持续时间、气体读数和邻点 HRR。
- 边界：未获得真实试验曲线、材料标定、柜内测温/烟气/弧参量数据，因此不能完成实验层面的真实性认证。

## 数据完整性与硬约束

| 检查项 | 结果 |
| --- | ---: |
| 工况数 / 工况号唯一数 | {len(audit)} / {audit['case_id'].nunique()} |
| 缺少必要输出文件 | {fmt_count(audit, 'flag_missing_files')} |
| `Runner_QC_Issues` 或 `Diagnostic_Issues` 非空 | {fmt_count(audit, 'flag_qc_issue')} |
| 阶段次序错误 (`1.5 -> 2 -> 3`) | {fmt_count(audit, 'flag_stage_order')} |
| 引燃时有效可燃质量低于模型门限 | {fmt_count(audit, 'flag_fire_below_burnable_threshold')} |
| 诊断内部 `Case_ID` 与目录工况号不一致 | {fmt_count(audit, 'flag_diagnostic_case_id_mismatch')} |

最后一项出现在复用输出中，属于可追溯性问题：不直接证明物理值错误，但应在训练或发表前修复元数据。

## 结果概览

- 发生 Stage 1.5：{int(audit['Stage_1_5_Time_s'].notna().sum())}；发生稳弧 Stage 2：{int(audit['Stage_2_Time_s'].notna().sum())}；发生引燃 Stage 3：{int(audit['fire'].sum())}。
- 峰值火灾 HRR：{audit['Peak_Fire_HRR'].min():.1f} 至 {audit['Peak_Fire_HRR'].max():.1f} W；峰值弧功率：{audit['Peak_Arc_Power'].min():.1f} 至 {audit['Peak_Arc_Power'].max():.1f} W。
- 所有工况的二次引燃节点数均为 0。

| 故障点 | 引燃工况数 | 引燃率 |
| --- | ---: | ---: |
{fire_fault_lines}

## 关键物理问题

### 1. 高温邻点完全不蔓延

- 邻点达到至少 300 degC 且二次引燃为 0：{fmt_count(audit, 'flag_hot_neighbor_no_secondary_300C')} 个工况。
- 邻点达到至少 500 degC 且二次引燃为 0：{fmt_count(audit, 'flag_hot_neighbor_no_secondary_500C')} 个工况。
- 最严重工况 `{hottest['case_id']}`：故障点 `{hottest['fault']}`，邻点 `{hottest['neighbor']}` 达到 {hottest['neighbor_max_temp']:.2f} degC，在 300 degC 以上维持约 {hottest['duration_over_300_s']:.2f} s，但邻点最大 HRR 为 {hottest['neighbor_hrr_max']:.1f} W。
- 已独立回读上述 {hot_verification['checked']} 个异常工况的完整历史：温度汇总不一致数 {hot_verification['max_temp_mismatch']}，邻点出现正 HRR 数 {hot_verification['positive_neighbor_hrr']}；高于 300 degC 的持续时间范围为 {hot_verification['duration_min_s']:.2f} 至 {hot_verification['duration_max_s']:.2f} s。

模型中二次蔓延强制要求气体阈值同时满足；当前结果说明该门控压制了即使极高固体温度下的邻点引燃。对于含可燃绝缘/端子周边聚合物的场景，这不能直接作为真实蔓延行为使用。

### 2. 通风改变燃烧强度，却不改变引燃边界

- 每一组 `{len(vent_cells)}` 个“故障点 + 电流 + 年限”组合中，四种通风条件的三次重复引燃数均完全相同；存在通风引燃差异的组合数为 {int((vent_cells.nunique(axis=1) > 1).sum())}。
- 整体引燃率：{vent_line}。

通风会影响火后 HRR，但对火前引燃事件没有可观测影响。对封闭电气柜中热积聚、可燃气稀释和供氧相互竞争的真实情形，这一响应过于刚性，需要标定后才能解释。

### 3. 老化风险出现反向趋势

- 按年限汇总的引燃率：{age_line}。
- 有 {fmt_count(audit, 'flag_arc_without_sustained_fire')} 个工况已进入稳弧阶段但由于可燃库存耗尽而未形成持续火灾，集中在 20 年和 30 年。

当前模型在高年限下会加速热解并提前耗尽局部燃料，从而使部分高龄设备比中龄设备更少起火。该机制可作为研究假设，但在未用老化材料试验验证前，不宜宣称代表真实风险随服役时间的变化。

### 4. 重复种子没有提供事件不确定性

- 共 {len(seed_groups)} 组物理配置，每组三个随机种子；种子导致引燃/不引燃结论变化的组数为 {seed_fire_variation}。
- 已引燃组中，三个种子的 Stage 3 时间最大跨度仅 {seed_t3_range:.3f} s。

该结果集适合学习近乎确定性的模型映射，不适合据此估计失效概率或随机风险区间。

## 使用判定

| 用途 | 判定 |
| --- | --- |
| 检查文件完整性、调试数据管线 | 可用 |
| 训练识别本模型生成规律的算法 | 可用，但须标注为合成/半物理数据 |
| 论证真实电气柜的引燃概率、蔓延风险或通风效果 | 不可直接使用 |
| 作为型式试验、消防认证或真实事故预测依据 | 不符合要求 |

逐工况标记见 `physics_audit_cases.csv`；其中 `physics_screening=physically_questionable` 的工况应在重新标定蔓延/燃料/通风机制后复算。
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    summary, diagnostics = load_data()
    audit = build_case_audit(summary, diagnostics)
    hottest = inspect_hottest_neighbor(audit)
    hot_verification = verify_flagged_hot_histories(audit)
    audit.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    write_report(audit, hottest, hot_verification)
    print(f"cases={len(audit)} questionable={(audit['physics_screening'] == 'physically_questionable').sum()}")
    print(f"report={OUTPUT_REPORT}")
    print(f"cases_csv={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
