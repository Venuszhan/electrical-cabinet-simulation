# Data Package Manifest

This repository is a compact public release package for a 1920-case multi-physics system accident dataset. It is designed for GitHub distribution: code, case definitions, summary tables, and small representative examples are included; large raw per-case training artifacts are excluded.

## Included

| Path | Contents |
| --- | --- |
| `multiphysics_accident_model/` | Accident modeling package and configuration tables. |
| `scripts/` | Reproducibility, calibration-demo, validation, and audit scripts. |
| `case_definitions/` | Complete 1920-case definitions and split manifest. |
| `results_summary/` | Case-level summary table. |
| `data_examples/accident_case_examples/` | Small accident-case examples copied from the original per-case outputs. |
| `data_examples/source_domain_examples/` | Initial calibration/source-domain time-series record and metadata. |
| `docs/` | Notes describing how external experimental/literature evidence constrains calibration. |

## Representative Accident-Case Examples

The example cases use the original per-case `system_timeseries.csv.gz` files.

Included cases:

| Case | Purpose |
| --- | --- |
| `CASE_00001` | Low-current no-fire example. |
| `CASE_00379` | KM1_DC ignition example. |
| `CASE_00427` | KM1_DC high-intensity ignition example. |
| `CASE_00907` | KM1_AC ignition example. |
| `CASE_01184` | X1 ignition example. |
| `CASE_01500` | X2 low-current no-fire example. |
| `CASE_01800` | X2 aged ignition example. |
| `CASE_01869` | X2 high-current ignition example. |

Each example case contains:

| File | Description |
| --- | --- |
| `case_config.json` | Case configuration. |
| `summary.json` | Per-case summary. |
| `diagnostics.csv` | Tabular diagnostic outputs. |
| `diagnostics.json` | Diagnostic metadata. |
| `system_timeseries.csv.gz` | Raw compressed system time series from the original case output. |

Large files such as `pigat_data.npz` and `history.csv.gz` are intentionally not included in this GitHub package.

## Excluded

| Excluded item | Reason |
| --- | --- |
| Full 1920-case raw output tree | Too large for a practical GitHub repository. |
| `CASE_*/pigat_data.npz` | Large downstream training artifact. |
| `CASE_*/history.csv.gz` | Large detailed history export; not needed for the compact public package. |
| Virtual environments | Reproducibility should use `requirements.txt`. |
| Logs, process IDs, Python cache, backup files | Runtime or local development artifacts. |

## Source-Domain Calibration Data

The repository includes `data_examples/source_domain_examples/initial_calibration_case_history.csv`, an initial calibration/source-domain time-series record used to demonstrate the transfer-calibration workflow. The duplicated Excel workbook is not included because the CSV contains the same tabular record in a more repository-friendly format.
