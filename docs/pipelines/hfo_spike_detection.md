# HFO/Spike Detection Pipeline

This page documents the `hfo_spike_detection` pipeline for development and internal reference. It is intentionally not linked from the main README or documentation index for releases that do not include this pipeline.

## Objective

The pipeline detects high-frequency oscillations and spikes in iEEG signals with a Derivative of Gaussian wavelet approach.

## Processing Summary

1. Load the primary iEEG file.
2. Select/exclude channels and apply the configured montage.
3. Run oscillation and/or spike detection over configured frequency bands.
4. Compute event counts and rates.
5. Write structured HDF5/MATLAB output or TSV event/rate tables.

## Algorithm Parameters

`HfoSpikeDetectorParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `alpha` | `0.005` | Significance level for automatic thresholding. |
| `detection_type` | `["Osc", "Spk"]` | Event families to detect: oscillations, spikes, or both. |
| `freq_band` | `[[80, 500]]` | Oscillation frequency bands as `[low, high]` pairs in Hz. |
| `thr_type` | `40.0` | `"auto"` or a numeric fixed threshold. |
| `param_thr` | `None` | Optional `[hfo_time_thr, hfo_freq_thr, spike_time_thr, spike_freq_thr]`. |
| `quantile_method` | `"hazen"` | Quantile method used by thresholding. |
| `nb_voices` | `12` | Voices per octave for the wavelet transform. |
| `vanishing_moment` | `20` | DoG mother wavelet vanishing moment. |
| `montage_mode` | `"bipolar"` | Channel referencing mode. |
| `bipolar_direction` | `"next_minus_previous"` | Subtraction direction for bipolar montage. |
| `bipolar_storage` | `"next_minus_previous"` | Naming/storage convention for bipolar channels. |
| `channels_for_montage` | `None` | Exact list, regex, or per-subject channel mapping. |
| `channels_to_exclude_for_montage` | `None` | Exact list or regex to exclude. |

## Writer Parameters

`HfoSpikeDetectorWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"hfo_spike_detection"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_description` | `"hfospikes"` | Value used in `desc-<value>`. |
| `output_suffix` | `"events"` | BIDS suffix. |
| `output_format` | `"hdf5"` | `hdf5`, `tsv`, or `matlab`. |

TSV mode writes event rows and per-channel rates.
