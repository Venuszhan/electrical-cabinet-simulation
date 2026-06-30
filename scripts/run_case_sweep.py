"""Run the 1920-case multi-physics accident sweep.

Parameter space, defined in ``multiphysics_accident_model/core/case_generator.py``:
    4 fault sites x 8 current levels x 5 service ages x 4 vents x 3 seeds = 1920.
Fixed controls: door_state=closed, ambient_temp=26.0, current_profile=constant.
Outputs are written to ``outputs/accident_cases_1920/<CASE_ID>/`` by default.
The runner can resume because cases with an existing ``diagnostics.json`` are skipped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "accident_cases_1920"
CASES_CSV = OUTPUT_DIR / "cases.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
LOG_FILE = OUTPUT_DIR / "run.log"

NODE_COLUMN_RE = re.compile(r"^(.+)_(V|T|T_Core|C|Char|HRR)$")


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def generate_cases(save_path: Path) -> pd.DataFrame:
    from multiphysics_accident_model.core.case_generator import generate_case_table

    df = generate_case_table()
    df = df.rename(columns={"random_seed": "seed"})
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


def sanitize_history(history_df: pd.DataFrame, sim) -> pd.DataFrame:
    clean = history_df.copy()
    if "Line_Current" in clean.columns:
        clean["Line_Current"] = clean["Line_Current"].ffill().bfill().fillna(0.0)
    if "Carb_Track_At_Stage15" in clean.columns:
        clean["Carb_Track_At_Stage15"] = clean["Carb_Track_At_Stage15"].fillna(0.0)
    for node in sim.nodes:
        defaults = {
            f"{node}_V": 0.0,
            f"{node}_T": sim.ambient_temp,
            f"{node}_T_Core": sim.ambient_temp,
            f"{node}_C": 0.0,
            f"{node}_Char": 0.0,
            f"{node}_HRR": 0.0,
        }
        for col, default in defaults.items():
            if col in clean.columns:
                clean[col] = clean[col].ffill().bfill().fillna(default)
    return clean


def get_neighbor_nodes(sim):
    neighbors = set()
    for node, _k_th in sim.solid_thermal_neighbors.get(sim.f_node, []):
        neighbors.add(node)
    for node, *_rest in sim.air_neighbors.get(sim.f_node, []):
        neighbors.add(node)
    for left, right in sim.copper_links:
        if left == sim.f_node:
            neighbors.add(right)
        elif right == sim.f_node:
            neighbors.add(left)
    neighbors.discard(sim.f_node)
    return sorted(neighbors)


def max_history_value(history_df: pd.DataFrame, nodes, suffix: str):
    values = []
    for node in nodes:
        column = f"{node}_{suffix}"
        if column in history_df.columns:
            series = pd.to_numeric(history_df[column], errors="coerce")
            if not series.dropna().empty:
                values.append(float(series.max()))
    return round(max(values), 3) if values else None


def build_spread_summary(sim, history_df: pd.DataFrame) -> dict:
    secondary_times = {
        node: time_s
        for node, time_s in sim.node_ignition_time.items()
        if node != sim.f_node and time_s is not None
    }
    secondary_nodes = sorted(secondary_times, key=lambda node: (secondary_times[node], node))
    neighbor_nodes = get_neighbor_nodes(sim)
    exposure_by_neighbor = {
        node: float(sim.spread_exposure.get(node, 0.0))
        for node in neighbor_nodes
    }
    top_risk_neighbor = None
    if exposure_by_neighbor:
        top_risk_neighbor = max(
            exposure_by_neighbor,
            key=lambda node: (
                exposure_by_neighbor[node],
                float(sim.states[node][0]),
                float(sim.states[node][1]),
                node,
            ),
        )
    return {
        "Secondary_Ignited_Count": len(secondary_nodes),
        "Secondary_Ignited_Nodes": ";".join(secondary_nodes),
        "Secondary_Ignition_Causes": ";".join(
            f"{node}:{sim.node_ignition_cause.get(node)}" for node in secondary_nodes
        ),
        "Top_Risk_Neighbor": top_risk_neighbor,
        "Max_Neighbor_T": max_history_value(history_df, neighbor_nodes, "T_Core"),
        "Max_Neighbor_C": max_history_value(history_df, neighbor_nodes, "C"),
    }


def system_only_columns(history_df: pd.DataFrame, sim):
    system_cols = []
    for col in history_df.columns:
        match = NODE_COLUMN_RE.match(col)
        if match and match.group(1) in sim.nodes:
            continue
        system_cols.append(col)
    return system_cols


def validate_case(history_df, diag, pigat_data, sim):
    issues = []
    if history_df.empty:
        issues.append("empty_history")
    if diag.get("Stage_3_Time_s") is not None and float(diag.get("Peak_Fire_HRR_W", 0.0)) <= 0.0:
        issues.append("stage3_with_zero_peak_fire_hrr")
    if diag.get("Stage_3_Time_s") is None:
        if float(diag.get("Peak_Fire_HRR_W", 0.0)) > 0.0 or int(diag.get("Nodes_Ignited", 0)) > 0:
            issues.append("fake_fire_without_stage3")
    if history_df.isna().any().any():
        issues.append("nan_in_history")
    if pigat_data is None:
        issues.append("missing_pigat_data")
    else:
        n_nodes = len(sim.nodes)
        if pigat_data["x_window"].shape[1] != n_nodes:
            issues.append(f"node_count_mismatch:x={pigat_data['x_window'].shape[1]} sim={n_nodes}")
        if pigat_data["edge_feat"].shape != (n_nodes, n_nodes, 4):
            issues.append(f"edge_feat_shape_invalid:{pigat_data['edge_feat'].shape}")
        for key, value in pigat_data.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                if not np.isfinite(value).all():
                    issues.append(f"nan_or_inf_in_pigat:{key}")
                    break
    if diag.get("Diagnostic_Issues"):
        issues.append(str(diag["Diagnostic_Issues"]))
    return issues


def assert_effective_config(sim, case: dict):
    expected = {
        "fault_terminal": str(case["fault_terminal"]),
        "current_level": float(case["current_level"]),
        "service_years": float(case["service_years"]),
        "vent_state": str(case["vent_state"]),
        "door_state": "closed",
        "ambient_temp": 26.0,
        "current_profile": "constant",
    }
    actual = {
        "fault_terminal": sim.fault_site,
        "current_level": sim.cur_I0,
        "service_years": sim.service_years,
        "vent_state": sim.vent_state,
        "door_state": sim.door_state,
        "ambient_temp": sim.ambient_temp,
        "current_profile": sim.current_profile_name,
    }
    mismatches = [
        f"{key}:expected={expected[key]!r},actual={actual[key]!r}"
        for key in expected
        if expected[key] != actual[key]
    ]
    if mismatches:
        raise ValueError("effective case configuration mismatch: " + "; ".join(mismatches))


def run_case(case: dict, case_dir: Path) -> dict:
    from multiphysics_accident_model.core.state import PIGAT_Realistic_DigitalTwin_V3
    from multiphysics_accident_model.observation.pigate_adapter import (
        convert_case_to_pigate,
        save_pigate_data,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case_config.json").open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(case), handle, ensure_ascii=False, indent=2)

    sim = PIGAT_Realistic_DigitalTwin_V3(
        init_current=float(case["current_level"]),
        fault_terminal=str(case["fault_terminal"]),
        service_days=30,
        service_years=float(case["service_years"]),
        seed=int(case["seed"]),
        case_config=case,
    )
    assert_effective_config(sim, case)

    history_df_raw, t1_5, t2, t3, q_at_ig, diag = sim.run()
    diag = make_jsonable(diag)
    history_df = sanitize_history(history_df_raw, sim)
    pigat_data = convert_case_to_pigate(sim, history_df, dt_record=sim.dt_record)
    save_pigate_data(pigat_data, str(case_dir / "pigat_data.npz"))

    # 直接写宽表，不做 long-format split（避免 pd.concat 内存翻倍）
    history_df.to_csv(case_dir / "history.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    sys_cols = system_only_columns(history_df, sim)
    history_df[sys_cols].to_csv(case_dir / "system_timeseries.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    issues = validate_case(history_df, diag, pigat_data, sim)
    diag["Runner_QC_Issues"] = "; ".join(issues) if issues else None
    diag["Runner_Node_Count"] = len(sim.nodes)
    diag["Runner_Edge_Feat_Shape"] = list(pigat_data["edge_feat"].shape) if pigat_data else None
    with (case_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diag, handle, ensure_ascii=False, indent=2)
    pd.DataFrame([diag]).to_csv(case_dir / "diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = {
        **case,
        "Stage_1_5_Time_s": diag.get("Stage_1_5_Time_s", t1_5),
        "Stage_2_Time_s": diag.get("Stage_2_Time_s", t2),
        "Stage_3_Time_s": diag.get("Stage_3_Time_s", t3),
        "Peak_Fire_HRR": diag.get("Peak_Fire_HRR_W"),
        "Peak_Arc_Power": diag.get("Peak_Arc_Power_W"),
        "Aging_Ea_kJ_per_mol": diag.get("Aging_Ea_kJ_per_mol"),
        "Aging_Ignition_Temp_Rise_C": diag.get("Aging_Ignition_Temp_Rise_C"),
        "Aging_Combustion_Rate_Multiplier": diag.get("Aging_Combustion_Rate_Multiplier"),
        "Fault_Burnable_Mass_at_Stage3_g": diag.get("Fault_Burnable_Mass_at_Stage3_g"),
        "Arc_Without_Sustained_Fire": diag.get("Arc_Without_Sustained_Fire"),
        **build_spread_summary(sim, history_df),
        "Diagnostic_Issues": diag.get("Diagnostic_Issues"),
        "Runner_QC_Issues": diag.get("Runner_QC_Issues"),
        "status": "success" if not issues else "qc_warning",
        "n_rows": len(history_df),
        "n_nodes": len(sim.nodes),
        "edge_feat_shape": "x".join(map(str, pigat_data["edge_feat"].shape)) if pigat_data else None,
    }
    return make_jsonable(summary)


SUMMARY_KEEP_COLUMNS = [
    "case_id", "fault_terminal", "current_level", "current_profile",
    "service_years", "ambient_temp", "vent_state", "door_state", "seed",
    "severity", "supply_mode", "circuit_id",
    "Stage_1_5_Time_s", "Stage_2_Time_s", "Stage_3_Time_s",
    "Peak_Fire_HRR", "Peak_Arc_Power",
    "Aging_Ea_kJ_per_mol", "Aging_Ignition_Temp_Rise_C", "Aging_Combustion_Rate_Multiplier",
    "Fault_Burnable_Mass_at_Stage3_g",
    "Arc_Without_Sustained_Fire",
    "Secondary_Ignited_Count", "Secondary_Ignited_Nodes", "Secondary_Ignition_Causes",
    "Top_Risk_Neighbor", "Max_Neighbor_T", "Max_Neighbor_C",
    "Diagnostic_Issues", "Runner_QC_Issues", "status",
    "n_rows", "n_nodes", "edge_feat_shape", "source_case_id",
]


def slim_summary(summary: dict) -> dict:
    return {col: summary.get(col) for col in SUMMARY_KEEP_COLUMNS}


def case_to_runtime_dict(row) -> dict:
    case = make_jsonable(row.to_dict())
    case["seed"] = int(case["seed"])
    return case


def worker(payload):
    import gc
    case, output_dir = payload
    case_dir = Path(output_dir) / case["case_id"]
    summary_path = case_dir / "summary.json"

    diag_path = case_dir / "diagnostics.json"
    if summary_path.exists() and diag_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            return slim_summary(cached), True
        except Exception:
            pass

    # 删除残留的失败痕迹（error.txt + 不完整文件）
    for stale in ("error.txt", "history.csv.gz", "node_timeseries.csv", "node_timeseries.csv.gz"):
        p = case_dir / stale
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    t0 = time.time()
    try:
        summary = run_case(case, case_dir)
        summary["wall_time_s"] = round(time.time() - t0, 2)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(make_jsonable(summary), fh, ensure_ascii=False, indent=2)
        gc.collect()
        return slim_summary(summary), False
    except Exception as exc:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        gc.collect()
        failed = {
            **case,
            "status": "failed",
            "Diagnostic_Issues": f"{type(exc).__name__}: {exc}",
            "Runner_QC_Issues": "exception",
            "wall_time_s": round(time.time() - t0, 2),
        }
        return slim_summary(failed), False


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    global OUTPUT_DIR, CASES_CSV, SUMMARY_CSV, LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16, help="parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="run only first N cases")
    parser.add_argument("--start", type=int, default=0, help="skip first N cases")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="directory for case outputs")
    parser.add_argument("--regenerate", action="store_true", help="regenerate cases.csv")
    parser.add_argument("--pending-only", action="store_true", help="process only cases without complete existing outputs")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir.resolve()
    CASES_CSV = OUTPUT_DIR / "cases.csv"
    SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
    LOG_FILE = OUTPUT_DIR / "run.log"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.regenerate or not CASES_CSV.exists():
        log(f"生成 cases.csv → {CASES_CSV}")
        cases_df = generate_cases(CASES_CSV)
    else:
        cases_df = pd.read_csv(CASES_CSV)
    log(f"总 case 数: {len(cases_df)}")

    if args.start:
        cases_df = cases_df.iloc[args.start:].copy()
    if args.limit is not None:
        cases_df = cases_df.head(args.limit).copy()
    if args.pending_only:
        completed_case_ids = {
            str(case_id)
            for case_id in cases_df["case_id"]
            if (
                (OUTPUT_DIR / str(case_id) / "summary.json").exists()
                and (OUTPUT_DIR / str(case_id) / "diagnostics.json").exists()
            )
        }
        cases_df = cases_df[~cases_df["case_id"].astype(str).isin(completed_case_ids)].copy()
        log(f"断点续跑跳过已完成 case 数: {len(completed_case_ids)}")
    log(f"本次将处理 {len(cases_df)} case，并行 workers={args.workers}")

    payloads = [(case_to_runtime_dict(row), str(OUTPUT_DIR)) for _, row in cases_df.iterrows()]

    # 流式写 summary，避免主进程囤积 3840 个 summary
    summary_path = SUMMARY_CSV
    write_header = not summary_path.exists()
    skipped = 0
    counts = {"success": 0, "qc_warning": 0, "failed": 0}
    t_start = time.time()
    with mp.Pool(processes=args.workers) as pool, summary_path.open("a", encoding="utf-8-sig", newline="") as csv_fh:
        import csv
        writer_csv = csv.DictWriter(csv_fh, fieldnames=SUMMARY_KEEP_COLUMNS)
        if write_header:
            writer_csv.writeheader()
        for idx, (summary, was_cached) in enumerate(pool.imap_unordered(worker, payloads, chunksize=1), start=1):
            writer_csv.writerow({k: ("" if summary.get(k) is None else summary.get(k)) for k in SUMMARY_KEEP_COLUMNS})
            csv_fh.flush()
            status = summary.get("status") or "failed"
            counts[status] = counts.get(status, 0) + 1
            if was_cached:
                skipped += 1
            elapsed = time.time() - t_start
            avg = elapsed / max(idx, 1)
            eta = avg * (len(payloads) - idx)
            if idx % 5 == 0 or idx == len(payloads):
                log(
                    f"进度 {idx}/{len(payloads)} "
                    f"last={summary.get('case_id')} status={status} "
                    f"counts={counts} "
                    f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min "
                    f"skipped={skipped}"
                )
            del summary
    log(f"完成。summary → {summary_path}, 总耗时 {(time.time()-t_start)/60:.1f} min")
    log(f"统计: {counts}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
