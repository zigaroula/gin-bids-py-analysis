from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.hdf5 import dataset_or_none, decode_str_array, str_scalar
from bidsforge.processing.utils.matlab import mat_root, mat_str, mat_str_list

from ..result import TFConditionEstimate, TFConditionPair
from .result import (
    TFConditionPredictorValues,
    TFConditionRegressionStats,
    TFPredictorValues,
    TFRegressionStats,
    TimeFrequencyRegressionResult,
)


def load_time_frequency_regression_result(path: Path | str) -> TimeFrequencyRegressionResult:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return _load_matlab(path)
    return _load_hdf5(path)


def _load_hdf5(path: Path) -> TimeFrequencyRegressionResult:
    with h5py.File(path, "r") as fh:
        _require_schema(fh, path.name)
        labels = decode_str_array(np.asarray(fh["meta/condition_labels"][:], dtype=object))
        cond_a, cond_b = labels[0], labels[1]
        source = str_scalar(dataset_or_none(fh, "provenance/source_tfr_file"), default=str(path))
        return TimeFrequencyRegressionResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(source)),
            channel_names=decode_str_array(np.asarray(fh["axes/channel"][:], dtype=object)),
            frequency_hz=np.asarray(fh["axes/frequency_hz"][:], dtype=np.float64),
            time_axis_s=np.asarray(fh["axes/time_s"][:], dtype=np.float64),
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
            regression=TFRegressionStats(
                condition_a=_load_condition_stats(fh, cond_a),
                condition_b=_load_condition_stats(fh, cond_b),
            ),
            predictor_values=TFPredictorValues(
                condition_a=_load_predictor_values(fh, cond_a),
                condition_b=_load_predictor_values(fh, cond_b),
            ),
            stats_valid=bool(np.asarray(fh["meta/stats_valid"][()]).item()),
            predictor=str_scalar(dataset_or_none(fh, "meta/predictor"), default="predictor_value"),
            predictor_zscore=str_scalar(dataset_or_none(fh, "meta/predictor_zscore"), default="none"),
            power_mode=str_scalar(dataset_or_none(fh, "meta/power_mode"), default="stored"),
            p_value_correction_method=str_scalar(
                dataset_or_none(fh, "meta/p_value_correction_method"),
                default="fdr_bh",
            ),
            significance_alpha=float(np.asarray(fh["meta/significance_alpha"][()]).item()),
        )


def _load_matlab(path: Path) -> TimeFrequencyRegressionResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    root = mat_root(mat, "time_frequency_regression")
    meta = root.meta
    labels = mat_str_list(meta.condition_labels)
    cond_a, cond_b = labels[0], labels[1]
    return TimeFrequencyRegressionResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(str(path))),
        metadata={"analysis_type": mat_str(getattr(meta, "analysis_type", None))},
        channel_names=mat_str_list(root.axes.channel),
        frequency_hz=np.asarray(root.axes.frequency_hz, dtype=np.float64).ravel(),
        time_axis_s=np.asarray(root.axes.time_s, dtype=np.float64).ravel(),
        condition_a=cond_a,
        condition_b=cond_b,
        condition_a_trial_count=int(np.asarray(meta.trial_counts).ravel()[0]),
        condition_b_trial_count=int(np.asarray(meta.trial_counts).ravel()[1]),
        signal_activity=TFConditionPair(
            condition_a=TFConditionEstimate(
                mean=np.asarray(getattr(root.data.signal_activity, cond_a).mean, dtype=np.float64),
                sem=np.asarray(getattr(root.data.signal_activity, cond_a).sem, dtype=np.float64),
            ),
            condition_b=TFConditionEstimate(
                mean=np.asarray(getattr(root.data.signal_activity, cond_b).mean, dtype=np.float64),
                sem=np.asarray(getattr(root.data.signal_activity, cond_b).sem, dtype=np.float64),
            ),
        ),
        regression=TFRegressionStats(
            condition_a=_load_matlab_condition_stats(
                getattr(root.stats.regression, cond_a),
                n_channels=len(mat_str_list(root.axes.channel)),
                n_freqs=int(np.asarray(root.axes.frequency_hz).size),
                n_times=int(np.asarray(root.axes.time_s).size),
            ),
            condition_b=_load_matlab_condition_stats(
                getattr(root.stats.regression, cond_b),
                n_channels=len(mat_str_list(root.axes.channel)),
                n_freqs=int(np.asarray(root.axes.frequency_hz).size),
                n_times=int(np.asarray(root.axes.time_s).size),
            ),
        ),
        predictor_values=TFPredictorValues(
            condition_a=_load_matlab_predictor_values(getattr(root.predictor, cond_a)),
            condition_b=_load_matlab_predictor_values(getattr(root.predictor, cond_b)),
        ),
        stats_valid=bool(np.asarray(meta.stats_valid).ravel()[0]),
        predictor=mat_str(getattr(meta, "predictor", None), default="predictor_value"),
        predictor_zscore=mat_str(getattr(meta, "predictor_zscore", None), default="none"),
        power_mode=mat_str(getattr(meta, "power_mode", None), default="stored"),
        p_value_correction_method=mat_str(
            getattr(meta, "p_value_correction_method", None),
            default="fdr_bh",
        ),
        significance_alpha=float(np.asarray(meta.significance_alpha).ravel()[0]),
    )


def _load_condition_stats(fh: h5py.File, condition: str) -> TFConditionRegressionStats:
    group = fh[f"stats/regression/{condition}"]
    return TFConditionRegressionStats(
        slope=np.asarray(group["slope"][:], dtype=np.float64),
        intercept=np.asarray(group["intercept"][:], dtype=np.float64),
        r_value=np.asarray(group["r_value"][:], dtype=np.float64),
        t_values=np.asarray(group["t_values"][:], dtype=np.float64),
        p_value=np.asarray(group["p_value"][:], dtype=np.float64),
        p_value_corrected=np.asarray(group["p_value_corrected"][:], dtype=np.float64),
        significant_mask=np.asarray(group["significant_mask"][:], dtype=bool),
        n_trials_used=int(np.asarray(group["n_trials_used"][()]).item()),
        stats_valid=bool(np.asarray(group["stats_valid"][()]).item()),
        permuted_slopes=(
            np.asarray(group["permuted_slopes"][:], dtype=np.float32)
            if "permuted_slopes" in group
            else None
        ),
    )


def _load_matlab_condition_stats(
    group: object,
    *,
    n_channels: int,
    n_freqs: int,
    n_times: int,
) -> TFConditionRegressionStats:
    return TFConditionRegressionStats(
        slope=_coerce_tf_map(group.slope, n_channels=n_channels, n_freqs=n_freqs, n_times=n_times),
        intercept=_coerce_tf_map(group.intercept, n_channels=n_channels, n_freqs=n_freqs, n_times=n_times),
        r_value=_coerce_tf_map(group.r_value, n_channels=n_channels, n_freqs=n_freqs, n_times=n_times),
        t_values=_coerce_tf_map(group.t_values, n_channels=n_channels, n_freqs=n_freqs, n_times=n_times),
        p_value=_coerce_tf_map(group.p_value, n_channels=n_channels, n_freqs=n_freqs, n_times=n_times),
        p_value_corrected=_coerce_tf_map(
            group.p_value_corrected,
            n_channels=n_channels,
            n_freqs=n_freqs,
            n_times=n_times,
        ),
        significant_mask=_coerce_tf_map(
            group.significant_mask,
            n_channels=n_channels,
            n_freqs=n_freqs,
            n_times=n_times,
            dtype=bool,
        ),
        n_trials_used=int(np.asarray(group.n_trials_used).ravel()[0]),
        stats_valid=bool(np.asarray(group.stats_valid).ravel()[0]),
        permuted_slopes=(
            np.asarray(group.permuted_slopes, dtype=np.float32)
            if hasattr(group, "permuted_slopes")
            else None
        ),
    )


def _load_predictor_values(fh: h5py.File, condition: str) -> TFConditionPredictorValues:
    group = fh[f"predictor/{condition}"]
    return TFConditionPredictorValues(
        raw_values=np.asarray(group["raw_values"][:], dtype=np.float64),
        transformed_values=np.asarray(group["transformed_values"][:], dtype=np.float64),
        values=np.asarray(group["values"][:], dtype=np.float64),
    )


def _load_matlab_predictor_values(group: object) -> TFConditionPredictorValues:
    return TFConditionPredictorValues(
        raw_values=np.asarray(group.raw_values, dtype=np.float64).ravel(),
        transformed_values=np.asarray(group.transformed_values, dtype=np.float64).ravel(),
        values=np.asarray(group.values, dtype=np.float64).ravel(),
    )


def _require_schema(fh: h5py.File, name: str) -> None:
    schema = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    if schema != "time_frequency_stats_subject":
        raise ValueError(f"{name}: unsupported schema_name={schema!r}.")


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
