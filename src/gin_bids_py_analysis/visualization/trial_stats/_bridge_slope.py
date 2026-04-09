"""Bridge between in-memory TrialSlopeStatsProcessingResult and slope-group processing.

``TrialSlopeStatsGroupProcessing.process_group()`` expects BIDS-like ``*_stats.h5``
files. This module builds an in-memory HDF5 file per subject matching the schema
consumed by ``trial_slope_stats_group.processor._load_raw_from_hdf5``, attaches
it to a synthetic ``BIDSFile``, and returns a compatible ``BIDSFileGroup``.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import TYPE_CHECKING, Generator
from uuid import uuid4

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.processing.trial_slope_stats_group import (
    build_trial_slope_stats_compatible_groups,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
    from gin_bids_py_analysis.processing.trial_slope_stats.result import (
        TrialSlopeStatsProcessingResult,
    )


class _MockPyBIDSFile:
    """Minimal stand-in for a pybids BIDSFile."""

    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _write_hdf5_structure(
    fh: h5py.File,
    result: "TrialSlopeStatsProcessingResult",
    subject_id: str,
) -> None:
    """Populate *fh* with datasets required by slope-group HDF5 reader."""
    del subject_id
    str_dt = h5py.string_dtype(encoding="utf-8")

    axes = fh.create_group("axes")
    primary_axis_name = "region" if result.analysis_level == "roi" else "channel"
    axes.create_dataset(
        primary_axis_name,
        data=np.array(result.channel_names, dtype=object),
        dtype=str_dt,
    )
    axes.create_dataset("time_s", data=result.time_axis_s.astype(np.float64))

    regression = fh.create_group("regression")
    cond_a = regression.create_group("condition_a")
    cond_b = regression.create_group("condition_b")
    cond_a.create_dataset("slope", data=np.asarray(result.condition_a_slope, dtype=np.float64))
    cond_b.create_dataset("slope", data=np.asarray(result.condition_b_slope, dtype=np.float64))
    cond_a.create_dataset("r_value", data=np.asarray(result.condition_a_r_value, dtype=np.float64))
    cond_b.create_dataset("r_value", data=np.asarray(result.condition_b_r_value, dtype=np.float64))

    means = fh.create_group("means")
    means.create_dataset(result.condition_a, data=np.asarray(result.condition_a_mean, dtype=np.float64))
    means.create_dataset(result.condition_b, data=np.asarray(result.condition_b_mean, dtype=np.float64))

    meta = fh.create_group("meta")
    meta.create_dataset("analysis_type", data=np.bytes_(str(result.analysis_type)))
    meta.create_dataset("analysis_level", data=np.bytes_(str(result.analysis_level)))
    meta.create_dataset(
        "trial_count_labels",
        data=np.array([result.condition_a, result.condition_b], dtype=object),
        dtype=str_dt,
    )
    binning_mode = (
        str(result.metadata.get("binning_mode"))
        if result.metadata.get("binning_mode") is not None
        else (
            "window_ms"
            if result.window_ms > 0
            else ("n_bins" if result.n_bins > 0 else "none")
        )
    )
    meta.create_dataset("binning_mode", data=np.bytes_(binning_mode))
    meta.create_dataset("window_ms", data=float(result.metadata.get("window_ms", result.window_ms)))
    meta.create_dataset("n_bins", data=int(result.metadata.get("n_bins", result.n_bins)))
    meta.create_dataset(
        "effective_n_bins",
        data=int(result.metadata.get("effective_n_bins", len(result.time_axis_s))),
    )
    meta.create_dataset("predictor", data=np.bytes_(str(result.predictor)))
    meta.create_dataset("predictor_zscore", data=np.bytes_(str(result.predictor_zscore)))
    meta.create_dataset(
        "predictor_transform_by_condition_json",
        data=np.bytes_(
            json.dumps(
                result.predictor_transform_by_condition,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )
    meta.create_dataset(
        "trial_activity_summary_kind",
        data=np.bytes_(str(result.trial_activity_summary_kind)),
    )
    meta.create_dataset(
        "trial_activity_summary_missing_response_policy",
        data=np.bytes_(str(result.trial_activity_summary_missing_response_policy)),
    )
    meta.create_dataset(
        "trial_activity_summary_source_json",
        data=np.bytes_(
            json.dumps(
                result.trial_activity_summary_source,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )
    meta.create_dataset(
        "trial_activity_summary_label",
        data=np.bytes_(str(result.trial_activity_summary_label)),
    )
    meta.create_dataset("activity_zscore", data=np.bytes_(str(result.activity_zscore)))
    meta.create_dataset(
        "activity_baseline_tmin_s",
        data=float(result.activity_baseline_tmin_s),
    )
    meta.create_dataset(
        "activity_baseline_tmax_s",
        data=float(result.activity_baseline_tmax_s),
    )

    prov = fh.create_group("provenance")
    prov.create_dataset(
        "source_ieeg_files",
        data=np.array(result.source_ieeg_files or [], dtype=object),
        dtype=str_dt,
    )
    prov.create_dataset(
        "source_electrodes_files",
        data=np.array(result.source_electrodes_files or [], dtype=object),
        dtype=str_dt,
    )

    # predictor values (required for scatter)
    pred_grp = fh.create_group("predictor")
    pred_grp.create_dataset(
        "condition_a_raw_values",
        data=np.asarray(result.condition_a_predictor_raw_values, dtype=np.float64),
    )
    pred_grp.create_dataset(
        "condition_b_raw_values",
        data=np.asarray(result.condition_b_predictor_raw_values, dtype=np.float64),
    )
    pred_grp.create_dataset(
        "condition_a_transformed_values",
        data=np.asarray(result.condition_a_predictor_transformed_values, dtype=np.float64),
    )
    pred_grp.create_dataset(
        "condition_b_transformed_values",
        data=np.asarray(result.condition_b_predictor_transformed_values, dtype=np.float64),
    )
    pred_grp.create_dataset(
        "condition_a_values",
        data=np.asarray(result.condition_a_predictor_values, dtype=np.float64),
    )
    pred_grp.create_dataset(
        "condition_b_values",
        data=np.asarray(result.condition_b_predictor_values, dtype=np.float64),
    )

    # epoch means (required for scatter)
    em_a = np.asarray(result.condition_a_epoch_means, dtype=np.float64)
    em_b = np.asarray(result.condition_b_epoch_means, dtype=np.float64)
    if em_a.ndim == 2 and em_a.size > 0 and em_b.ndim == 2 and em_b.size > 0:
        scatter_grp = fh.create_group("scatter")
        scatter_grp.create_dataset("condition_a_epoch_means", data=em_a)
        scatter_grp.create_dataset("condition_b_epoch_means", data=em_b)

    summary_a = np.asarray(result.condition_a_trial_activity_summary_values, dtype=np.float64)
    summary_b = np.asarray(result.condition_b_trial_activity_summary_values, dtype=np.float64)
    if summary_a.ndim == 2 and summary_b.ndim == 2:
        summary_grp = fh.create_group("trial_activity_summary")
        summary_grp.create_dataset("condition_a_values", data=summary_a)
        summary_grp.create_dataset("condition_b_values", data=summary_b)
        summary_grp.create_dataset("kind", data=np.bytes_(str(result.trial_activity_summary_kind)))
        summary_grp.create_dataset(
            "missing_response_policy",
            data=np.bytes_(str(result.trial_activity_summary_missing_response_policy)),
        )
        summary_grp.create_dataset(
            "source_json",
            data=np.bytes_(
                json.dumps(
                    result.trial_activity_summary_source,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )
        summary_grp.create_dataset("label", data=np.bytes_(str(result.trial_activity_summary_label)))


def _build_in_memory_bids_file_with_handle(
    result: "TrialSlopeStatsProcessingResult",
    subject_id: str,
) -> tuple[BIDSFile, h5py.File]:
    fake_path = Path(f"/in-memory/sub-{subject_id}_{uuid4().hex}_stats.h5")
    primary = result.source_group.primary
    entities = {
        "subject": subject_id,
        "task": str(primary.get("task") or ""),
        "desc": str(primary.get("desc") or ""),
        "suffix": "stats",
        "extension": ".h5",
    }
    mock = _MockPyBIDSFile(path=str(fake_path), entities=entities)
    bids_file = BIDSFile(mock)
    fh: h5py.File = h5py.File(str(fake_path), "w", driver="core", backing_store=False)
    _write_hdf5_structure(fh, result, subject_id)
    bids_file.attach_data(fh)
    return bids_file, fh


@contextmanager
def group_file_group_context_slope(
    results: dict[str, "TrialSlopeStatsProcessingResult"],
) -> Generator["BIDSFileGroup", None, None]:
    """Build a slope-group compatible in-memory BIDSFileGroup and close handles."""
    if not results:
        raise ValueError("group_file_group_context_slope requires at least one result.")

    handles: list[h5py.File] = []
    try:
        bids_files: list[BIDSFile] = []
        for subject_id, result in sorted(results.items()):
            bids_file, fh = _build_in_memory_bids_file_with_handle(result, subject_id)
            handles.append(fh)
            bids_files.append(bids_file)

        groups = build_trial_slope_stats_compatible_groups(bids_files)
        if not groups:
            raise ValueError(
                "build_trial_slope_stats_compatible_groups returned no groups for the provided results."
            )
        yield max(groups, key=lambda g: len(g.all_files))
    finally:
        for fh in handles:
            try:
                if fh.id.valid:
                    fh.close()
            except Exception:  # noqa: BLE001
                pass
