# Time-Frequency Regression

`time_frequency_regression` computes per-condition trial-wise regression maps
from stored `time_frequency` derivatives. It targets the regression branch of
the MATLAB `b2_TF_subj_contact_levels_CB*.m` scripts.

## Processing Summary

For each subject group, the processor:

1. loads a `time_frequency` `.h5` or `.mat` derivative;
2. maps stored TFR trials to behavior rows with a `TableTrialResolver`;
3. reads a continuous predictor from resolved-trial metadata;
4. optionally applies per-condition affine transforms and predictor z-scoring;
5. fits OLS maps per condition over `channel x frequency x time`;
6. writes slope, intercept, r, t, p, corrected p, and significance masks.

The output maps use:

```text
channel x frequency x time
```

See `scripts/run_time_frequency_regression.py` for an editable runner.

## Downstream Group Analysis

Use `time_frequency_regression_group` to aggregate channel-level subject outputs
into ROI maps. The default group metric is `t_values`, matching the Matlab b3
TF regression display logic; `slope` and `r_value` can also be aggregated.

See [Time-Frequency Regression Group](./time_frequency_regression_group.md).
