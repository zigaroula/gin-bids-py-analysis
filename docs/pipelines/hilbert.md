# Hilbert Pipeline

The `hilbert` pipeline extracts band-limited envelopes from continuous iEEG recordings. It can write compact HDF5/MATLAB derivatives or BrainVision files that can be used by downstream trial-statistics pipelines.

## Objective

Use this pipeline to transform raw iEEG into a time series of analytic signal envelopes over a frequency grid. Typical use cases include broadband band envelope activity extraction before trial-locked statistics.

## Processing Summary

1. Load the primary iEEG file with MNE.
2. Optionally read events from annotations or a matching `_events.tsv`.
3. Optionally apply a notch filter.
4. Select/exclude channels and optionally build a bipolar montage.
5. Build frequency bins from `f_min`, `f_max`, and `f_step`.
6. Band-pass filter each subband and compute Hilbert envelopes.
7. Average subbands, downsample, normalize, and smooth according to `method`.
8. Write a BIDS derivative under `derivatives/hilbert/`.

## Minimal Example

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)

ds = BIDSDataset("D:/data/bids")
groups = ds.get_files(scope="raw", suffix="ieeg", extension=".vhdr")

params = HilbertParams(
    f_min=50,
    f_max=150,
    f_step=10,
    smoothing_windows_ms=[0, 250],
)
writer = HilbertProcessingWriter(
    HilbertWriterParams(
        bids_root=ds.root,
        output_format="hdf5",
        output_description="bandenv",
    )
)

HilbertProcessing(params).run(groups, writer, n_jobs=1, skip_existing=True)
```

## Algorithm Parameters

`HilbertParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `f_min` | required | Lowest frequency-bin edge in Hz. |
| `f_max` | required | Highest frequency-bin edge in Hz. Must be greater than `f_min`. |
| `f_step` | required | Frequency-bin step in Hz. Adjacent bin pairs define subbands. |
| `computation_frequency_hz` | `None` | Optional resampling rate before filtering/Hilbert computation. |
| `downsampled_frequency_hz` | `64.0` | Output envelope sampling rate. `None` keeps the source rate. |
| `event_sample_shift_samples` | `0` | Offset applied to event samples before export. |
| `events_source` | `"annotations"` | One of `annotations`, `events_tsv`, or `auto`. |
| `notch_filter_freqs` | `[]` | Frequencies in Hz to notch before envelope extraction. |
| `montage_mode` | `"mono"` | `mono` or `bipolar`. |
| `bipolar_direction` | `"next_minus_previous"` | Subtraction direction for bipolar montage. |
| `bipolar_storage` | `"previous"` | Naming/storage convention for bipolar channels. |
| `channels_for_montage` | `None` | Exact channel list or regex to keep. |
| `channels_to_exclude_for_montage` | `None` | Exact channel list or regex to exclude. |
| `smoothing_windows_ms` | `[0]` | Sliding-average windows; `0` means no smoothing. |
| `normalization_mode` | `"percent"` | `none`, `percent`, `percent_centered`, or `db`. |
| `method` | `"localizer"` | `localizer` downsamples before normalization/smoothing; `spm2env` normalizes/smooths at native rate then downsamples. |

Important validation:

- `f_max` must be greater than `f_min`.
- `f_step` must not exceed `f_max - f_min`.
- Frequency bins above Nyquist are clamped during processing.

## Writer Parameters

`HilbertWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"hilbert"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder for outputs. |
| `output_suffix` | `"ieeg"` | BIDS suffix. |
| `output_description` | `"hilbert"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5`, `brainvision`, or `matlab`. |

Output formats:

- `hdf5`: one `.h5` file per input recording.
- `matlab`: one `.mat` file per input recording.
- `brainvision`: one `.vhdr/.vmrk/.eeg` triplet per smoothing window.

With BrainVision output, the smoothing window is appended to the description. For example, `output_description="bandenv"` and `smoothing_windows_ms=[0, 250]` produce descriptions such as `desc-bandenvsm0` and `desc-bandenvsm250`.

## Outputs

Expected output shape:

```text
derivatives/hilbert/
|-- dataset_description.json
`-- sub-01/
    `-- ieeg/
        `-- sub-01_task-example_desc-bandenv_ieeg.h5
```

The derivative `dataset_description.json` records the pipeline label and processing parameters under `GeneratedBy`.
