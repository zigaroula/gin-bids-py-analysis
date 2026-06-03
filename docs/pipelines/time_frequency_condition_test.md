# Time-Frequency Condition Test

`time_frequency_condition_test` computes subject-level condition contrasts from
stored `time_frequency` derivatives. It is the Python counterpart of the
condition-test branch of MATLAB `b2_TF_subj_contact_levels.m`.

## Processing Summary

For each subject group, the processor:

1. loads a `time_frequency` `.h5` or `.mat` derivative;
2. selects stored, raw, or baseline-corrected TF power;
3. optionally applies MATLAB-style grand-average baseline subtraction;
4. maps stored TFR trials to behavior rows with a `TableTrialResolver`;
5. optionally keeps a strict MATLAB time window (`t > min & t < max`);
6. computes condition means, SEM, difference, t-values, p-values, and masks.

The output maps use:

```text
channel x frequency x time
```

See `scripts/run_time_frequency_condition_test.py` for an editable runner.

## Downstream Group Analysis

Use `time_frequency_condition_test_group` to aggregate channel-level subject
outputs into ROI maps. The default group metric is `t_values`, matching the
Matlab b3 TF condition-test display logic.

See [Time-Frequency Condition Test Group](./time_frequency_condition_test_group.md).
