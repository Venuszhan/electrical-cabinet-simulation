# Transfer Calibration Notes

This document describes how external experimental or literature evidence can be used to constrain the multi-physics accident model. It is written as a general release note for public reuse rather than as a machine-specific project log.

## Scope

External source-domain evidence may be used to constrain qualitative trends, parameter ranges, or calibration priors. It should not be presented as a complete substitute for cabinet-specific experimental measurements unless such measurements are actually available and documented.

For this dataset, the public package provides:

- modeling code and configuration tables;
- complete 1920-case definitions;
- case-level summary tables;
- representative accident-case examples;
- documentation of the transfer-calibration assumptions.

The complete source-domain experimental dataset is not included in this compact GitHub package.

## Transfer Constraints

When using external thermal-aging, cable-burning, or insulation-reliability evidence, the constraints should be interpreted conservatively:

1. Aging may increase electrical fault likelihood through contact resistance, looseness, or oxidation.
2. Aging evidence from cable or polymer tests should not be directly treated as cabinet ignition probability without a mapping step.
3. External heat-release-rate evidence can constrain trend direction or plausible ranges, but not exact cabinet-specific HRR parameters by itself.
4. Char residue and post-ignition combustion effects should be separated from first-ignition criteria.

## Recommended Wording

Use wording such as:

> The accident-evolution model was calibrated with transfer constraints derived from external experimental and literature evidence. Representative source-domain examples or metadata are provided where release is permitted; the complete source-domain dataset is not included in this compact public package.

## Reuse Notes

Users adapting this repository should document:

- source-domain data provenance;
- whether the source-domain data are experimental, literature-derived, synthetic, or simulated;
- preprocessing steps;
- calibration targets and parameters;
- limitations of transferring evidence across test conditions.
