# Source-Domain Calibration Data

This directory contains the initial calibration/source-domain record used to demonstrate the transfer-calibration workflow.

Included file:

| File | Description |
| --- | --- |
| `initial_calibration_case_history.csv` | Initial calibration time-series record in CSV format. |
| `initial_calibration_case_metadata.json` | Metadata describing the calibration record and file schema. |

The CSV file uses the same column convention as the case time-series exports, including global case fields, electrical fault variables, stage labels, thermal/smoke variables, and node-level measurements.

Example usage:

```python
import pandas as pd

df = pd.read_csv("data_examples/source_domain_examples/initial_calibration_case_history.csv")
print(df[["Time", "Stage", "Fault_Terminal", "Line_Current", "Fire_HRR_Total"]].head())
```
