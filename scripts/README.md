# Utility Scripts

This directory contains small, editable recipes for running the public
`bidsforge` pipelines on a local BIDS dataset. They are meant as starting
points: copy a script, edit the configuration block at the top, and run it from
the repository root.

These scripts do not define a stable command-line interface. For reusable
projects, prefer keeping your own project-specific scripts outside the package
or converting the edited recipe into tested application code.

## Hilbert

Extract Hilbert-band envelopes from raw iEEG recordings:

```bash
python scripts/run_hilbert.py
```

## Condition Test

Compare two trial conditions at subject level:

```bash
python scripts/run_condition_test.py
```

## Regression

Compute condition-specific trial-wise regression maps at subject level:

```bash
python scripts/run_regression.py
```

## Condition Test Group

Aggregate subject-level `condition_test` outputs into group-level ROI
statistics:

```bash
python scripts/run_condition_test_group.py
```

## Regression Group

Aggregate subject-level `regression` outputs into group-level ROI statistics:

```bash
python scripts/run_regression_group.py
```

## Typical Order

```text
raw iEEG
  -> run_hilbert.py
  -> run_condition_test.py or run_regression.py
  -> run_condition_test_group.py or run_regression_group.py
```

All scripts use placeholder paths and generic trial-table columns. Edit
`BIDS_ROOT`, filters, event codes, condition rules, and ROI definitions before
running them on your data.
