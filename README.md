# Multi-physics system accident dataset

This repository contains a compact public release package for a 1920-case multi-physics system accident dataset. It includes the modeling code, complete case definitions, summary tables, and representative accident-case examples.

The package is designed for GitHub distribution. It does not include the full raw output tree or large downstream training artifacts.

## Repository Contents

```text
.
├── multiphysics_accident_model/                         # Accident modeling package
├── scripts/                           # Reproduction, validation, and audit scripts
├── case_definitions/                  # Complete 1920-case definitions
├── results_summary/                   # Case-level summary tables
├── data_examples/
│   ├── accident_case_examples/           # Representative accident-case records
│   └── source_domain_examples/        # Placeholder for permitted source-domain examples
├── docs/                              # Calibration and transfer notes
├── DATA_PACKAGE_MANIFEST.md
└── requirements.txt
```

## Data Included

`case_definitions/` contains the full 1920-case design table and split manifest.

`results_summary/` contains the case-level summary table.

`data_examples/accident_case_examples/` contains selected accident-case examples. Each example includes:

- `case_config.json`
- `summary.json`
- `diagnostics.csv`
- `diagnostics.json`
- `system_timeseries.csv.gz`

These examples are copied from the original case outputs.

## Data Not Included

The full raw per-case output tree is not included because it is too large for a practical GitHub repository. Large per-case files such as `pigat_data.npz` and `history.csv.gz` are also excluded from this compact release.

`data_examples/source_domain_examples/` contains the initial calibration/source-domain time-series record used to demonstrate the transfer-calibration workflow.

## Installation

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run a Small Demo

Run one calibration-style demonstration case:

```bash
python scripts/run_calibration_case.py
```

Run a limited case sweep:

```bash
python scripts/run_case_sweep.py --limit 5
```

The default full sweep writes outputs to:

```text
outputs/accident_cases_1920/
```

## Inspect Included Examples

Example time-series data are gzip-compressed CSV files. In Python:

```python
import pandas as pd

df = pd.read_csv("data_examples/accident_case_examples/CASE_00001/system_timeseries.csv.gz")
print(df.head())
```

Initial calibration/source-domain data can be loaded with:

```python
source_df = pd.read_csv("data_examples/source_domain_examples/initial_calibration_case_history.csv")
print(source_df.head())
```

## Citation

If you use this dataset or code, cite the associated paper or dataset DOI when available. Until a DOI is assigned, cite this repository and include the commit hash used for reproduction.

## License

Code is released under the MIT License. Data tables, representative examples, and documentation are released under CC BY 4.0 unless otherwise noted. See `LICENSE` for details.
