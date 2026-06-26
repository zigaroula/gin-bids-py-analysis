# Time-Frequency

The `time_frequency` pipeline computes trial-level multitaper time-frequency
power from raw BIDS iEEG recordings. It extracts epochs, computes DPSS
multitaper power, converts power to dB, computes a baseline, and writes a BIDS
derivative.

The spectral computation is selected with `method`. The default and currently
supported method is `method="fieldtrip"`, which follows FieldTrip
`ft_specest_mtmconvol` conventions. Baseline handling is intentionally separate:
`method="fieldtrip"` controls the spectral estimator, while `apply_baseline`
only controls whether the stored `power_db` is raw dB or baseline-corrected dB.

## Processing Summary

For each input recording, the processor:

1. resolves anchor events from annotations or a matching `_events.tsv`;
2. extracts epochs around `anchor_event_codes`;
3. optionally selects channels and applies a mono/bipolar montage;
4. evaluates TFR samples every `time_decimation` epoch samples;
5. computes DPSS multitaper power on the FieldTrip-aligned frequency grid;
6. converts power to `10 * log10(power)`;
7. computes `baseline_db` over `baseline_window_s`;
8. subtracts `baseline_db` from `power_db` when `apply_baseline=True`.

## Output Shape

The stored data order is:

```text
power_db:    trial x channel x frequency x time
baseline_db: trial x channel x frequency
```

`baseline_db` is always computed and written. The `apply_baseline` parameter
only controls whether this baseline is subtracted from `power_db` before
writing. The default is `apply_baseline=True`.

## Key Parameters

- `time_decimation=20`: evaluate one TFR time sample every 20 epoch samples.
- `frequency_start_hz=4`, `frequency_exponent_step=0.1`,
  `frequency_exponent_max=5.7`: define an exponential frequency grid.
- `low_frequency_cutoff_hz=32`: boundary between low- and high-frequency rules.
- `low_frequency_n_cycles=6`: low-frequency window duration is `6 / f`.
- `high_frequency_window_s=0.1875`: fixed high-frequency window duration.
- `baseline_window_s=(-1.3, -0.7)`: baseline window in TFR time coordinates.
- `apply_baseline=True`: write baseline-corrected `power_db` while also storing
  `baseline_db`.
- `method="fieldtrip"`: use the FieldTrip-compatible estimator.
- `fieldtrip_polyorder=0`: demean each trial/channel before spectral
  estimation, matching FieldTrip default polynomial removal.
- `fieldtrip_pad_s=None`: use epoch duration as the FFT pad length, matching
  FieldTrip when `pad` is omitted.
- `fieldtrip_padtype="zero"`: zero padding; other FieldTrip pad types are not
  implemented yet.

## FieldTrip Mode

`method="fieldtrip"` reproduces the important `ft_specest_mtmconvol` details:

- time windows use `round(timwin * fs)` without forcing odd window lengths;
- requested times are snapped with `unique(round(timeoi * fs) / fs)`;
- frequency bins are snapped to the FFT grid with
  `round(freqoi / (fs / endnsample)) + 1`;
- DPSS tapers are generated with `NW=timwinsample * tapsmofrq / fs`, then the
  last taper is discarded as FieldTrip does;
- wavelets are padded to `endnsample`, FFT-transformed, multiplied by the data
  spectrum, inverse-transformed, and `fftshift`ed before sampling;
- edge samples are masked with FieldTrip's valid-time rule;
- complex coefficients are scaled by `sqrt(2 / timwinsample)` before averaging
  taper power.

The derivative metadata records the selected method and effective FieldTrip
settings, including `time_frequency_method`, `fieldtrip_polyorder`,
`fieldtrip_pad_s`, `fieldtrip_padtype`, and `fieldtrip_scaling`.

## Example

See `scripts/run_time_frequency.py` for an editable runner. The default writer
uses HDF5 and writes under:

```text
derivatives/time_frequency/
```

## Performance Notes

The processor can run across files with `BaseProcessing.run(..., n_jobs=N)`.
When `pyfftw` is installed, its cache is enabled and the processor sets
`pyfftw.config.NUM_THREADS` using the shared `get_threads_for_worker()` helper,
so FFT threads do not multiply uncontrollably across joblib workers.
