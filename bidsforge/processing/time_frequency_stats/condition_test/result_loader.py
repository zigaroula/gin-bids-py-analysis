from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.hdf5 import dataset_or_none, decode_str_array, str_scalar
from bidsforge.processing.utils.matlab import mat_root, mat_str, mat_str_list

from ..result import TFConditionEstimate, TFConditionPair
from .result import (
    TFConditionContrast,
    TFDifferenceEstimate,
    TFGrandAverage,
    TimeFrequencyConditionTestResult,
)


def load_time_frequency_condition_test_result(
    path: Path | str,
) -> TimeFrequencyConditionTestResult:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return _load_matlab(path)
    return _load_hdf5(path)


def _load_hdf5(path: Path) -> TimeFrequencyConditionTestResult:
    with h5py.File(path, "r") as fh:
        _require_schema(fh, path.name)
        channels = decode_str_array(np.asarray(fh["axes/channel"][:], dtype=object))
        freqs = np.asarray(fh["axes/frequency_hz"][:], dtype=np.float64)
        times = np.asarray(fh["axes/time_s"][:], dtype=np.float64)
        labels = decode_str_array(np.asarray(fh["meta/condition_labels"][:], dtype=object))
        cond_a, cond_b = labels[0], labels[1]
        grand = None
        if "stats/grand_average" in fh:
            gg = fh["stats/grand_average"]
            grand = TFGrandAverage(
                mean=np.asarray(gg["mean"][:], dtype=np.float64),
                std=np.asarray(gg["std"][:], dtype=np.float64),
                t_values=np.asarray(gg["t_values"][:], dtype=np.float64),
                p_values=np.asarray(gg["p_values"][:], dtype=np.float64),
            )
        source = str_scalar(dataset_or_none(fh, "provenance/source_tfr_file"), default=str(path))
        return TimeFrequencyConditionTestResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(source)),
            metadata=_load_meta(fh),
            channel_names=channels,
            frequency_hz=freqs,
            time_axis_s=times,
            condition_a=cond_a,
            condition_b=cond_b,
            condition_a_trial_count=int(np.asarray(fh["meta/trial_counts"][:]).ravel()[0]),
            condition_b_trial_count=int(np.asarray(fh["meta/trial_counts"][:]).ravel()[1]),
            source_tfr_file=source,
            source_ieeg_files=_load_optional_str_list(fh, "provenance/source_ieeg_files"),
            source_electrodes_files=_load_optional_str_list(
                fh,
                "provenance/source_electrodes_files",
            ),
            signal_activity=TFConditionPair(
                condition_a=TFConditionEstimate(
                    mean=np.asarray(fh[f"data/signal_activity/{cond_a}/mean"][:], dtype=np.float64),
                    sem=np.asarray(fh[f"data/signal_activity/{cond_a}/sem"][:], dtype=np.float64),
                ),
                condition_b=TFConditionEstimate(
                    mean=np.asarray(fh[f"data/signal_activity/{cond_b}/mean"][:], dtype=np.float64),
                    sem=np.asarray(fh[f"data/signal_activity/{cond_b}/sem"][:], dtype=np.float64),
                ),
            ),
            stats_valid=bool(np.asarray(fh["meta/stats_valid"][()]).item()),
            difference=TFDifferenceEstimate(
                mean=np.asarray(fh["data/signal_activity/difference/mean"][:], dtype=np.float64),
                sem=np.asarray(fh["data/signal_activity/difference/sem"][:], dtype=np.float64),
            ),
            contrast=TFConditionContrast(
                t_values=np.asarray(fh["stats/condition_contrast/t_values"][:], dtype=np.float64),
                p_values=np.asarray(fh["stats/condition_contrast/p_values"][:], dtype=np.float64),
                p_values_uncorrected=np.asarray(
                    fh["stats/condition_contrast/p_values_uncorrected"][:],
                    dtype=np.float64,
                ),
                significant_mask=np.asarray(
                    fh["stats/condition_contrast/significant_mask"][:],
                    dtype=bool,
                ),
                permuted_t_values=(
                    np.asarray(fh["stats/condition_contrast/permuted_t_values"][:], dtype=np.float32)
                    if "stats/condition_contrast/permuted_t_values" in fh
                    else None
                ),
            ),
            grand_average=grand,
            power_mode=str_scalar(dataset_or_none(fh, "meta/power_mode"), default="stored"),
            p_value_correction_method=str_scalar(
                dataset_or_none(fh, "meta/p_value_correction_method"),
                default="fdr_bh",
            ),
            significance_alpha=float(np.asarray(fh["meta/significance_alpha"][()]).item()),
        )


def _load_matlab(path: Path) -> TimeFrequencyConditionTestResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    root = mat_root(mat, "time_frequency_condition_test")
    meta = root.meta
    channels = mat_str_list(root.axes.channel)
    labels = mat_str_list(meta.condition_labels)
    cond_a, cond_b = labels[0], labels[1]
    n_channels = len(channels)
    n_freqs = int(np.asarray(root.axes.frequency_hz).size)
    n_times = int(np.asarray(root.axes.time_s).size)
    return TimeFrequencyConditionTestResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(str(path))),
        metadata={"analysis_type": mat_str(getattr(meta, "analysis_type", None))},
        channel_names=channels,
        frequency_hz=np.asarray(root.axes.frequency_hz, dtype=np.float64).ravel(),
        time_axis_s=np.asarray(root.axes.time_s, dtype=np.float64).ravel(),
        condition_a=cond_a,
        condition_b=cond_b,
        signal_activity=TFConditionPair(
            condition_a=TFConditionEstimate(
                mean=_coerce_tf_map(
                    getattr(root.data.signal_activity, cond_a).mean,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                    n_times=n_times,
                ),
                sem=_coerce_tf_map(
                    getattr(root.data.signal_activity, cond_a).sem,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                    n_times=n_times,
                ),
            ),
            condition_b=TFConditionEstimate(
                mean=_coerce_tf_map(
                    getattr(root.data.signal_activity, cond_b).mean,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                    n_times=n_times,
                ),
                sem=_coerce_tf_map(
                    getattr(root.data.signal_activity, cond_b).sem,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                    n_times=n_times,
                ),
            ),
        ),
        difference=TFDifferenceEstimate(
            mean=_coerce_tf_map(
                root.data.signal_activity.difference.mean,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            ),
            sem=_coerce_tf_map(
                root.data.signal_activity.difference.sem,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            ),
        ),
        contrast=TFConditionContrast(
            t_values=_coerce_tf_map(
                root.stats.condition_contrast.t_values,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            ),
            p_values=_coerce_tf_map(
                root.stats.condition_contrast.p_values,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            ),
            p_values_uncorrected=_coerce_tf_map(
                root.stats.condition_contrast.p_values_uncorrected,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            ),
            significant_mask=_coerce_tf_map(
                root.stats.condition_contrast.significant_mask,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
                dtype=bool,
            ),
        ),
    )


def _require_schema(fh: h5py.File, name: str) -> None:
    schema = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    if schema != "time_frequency_stats_subject":
        raise ValueError(f"{name}: unsupported schema_name={schema!r}.")


def _load_meta(fh: h5py.File) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, node in fh["meta"].items():
        if isinstance(node, h5py.Dataset) and node.shape == ():
            value = node[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            out[key] = value
    return out


def _load_optional_str_list(fh: h5py.File, key: str) -> list[str]:
    ds = dataset_or_none(fh, key)
    if ds is None:
        return []
    return decode_str_array(np.asarray(ds[:], dtype=object))


def _coerce_tf_map(
    value: object,
    *,
    n_channels: int,
    n_freqs: int,
    n_times: int,
    dtype: object = np.float64,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    expected = (n_channels, n_freqs, n_times)
    if arr.shape == expected:
        return arr
    if n_channels == 1 and arr.shape == (n_freqs, n_times):
        return arr.reshape(expected)
    if n_freqs == 1 and arr.shape == (n_channels, n_times):
        return arr.reshape(expected)
    if n_times == 1 and arr.shape == (n_channels, n_freqs):
        return arr.reshape(expected)
    if arr.size == n_channels * n_freqs * n_times:
        return arr.reshape(expected)
    return np.full(expected, np.nan, dtype=dtype)
