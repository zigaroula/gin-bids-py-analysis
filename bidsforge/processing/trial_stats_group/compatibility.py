"""Compatibility helpers for group-level trial statistics inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.utils.channels import normalize_channel_name


@dataclass(frozen=True)
class SubjectStatsSignature:
    """Compatibility signature for one loaded subject-level result."""

    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    time_axis_hash: str
    time_axis_len: int
    analysis_level: str
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    activity_zscore: str
    activity_baseline_tmin_s: float
    activity_baseline_tmax_s: float
    activity_baseline_scope: str
    activity_baseline_remove_outlier_trial_means: bool
    trial_activity_summary_kind: str
    trial_activity_summary_missing_response_policy: str
    trial_activity_summary_source_json: str
    trial_activity_summary_label: str
    extra_key_parts: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.source_desc,
            self.condition_labels,
            self.time_axis_hash,
            self.time_axis_len,
            self.analysis_level,
            self.binning_mode,
            self.window_ms,
            self.n_bins,
            self.effective_n_bins,
            self.activity_zscore,
            self.activity_baseline_tmin_s,
            self.activity_baseline_tmax_s,
            self.activity_baseline_scope,
            self.activity_baseline_remove_outlier_trial_means,
            self.trial_activity_summary_kind,
            self.trial_activity_summary_missing_response_policy,
            self.trial_activity_summary_source_json,
            self.trial_activity_summary_label,
            self.extra_key_parts,
        )


@dataclass(frozen=True)
class SubjectStatsInput:
    """Technical wrapper around one loaded subject-level result."""

    stats_file: BIDSFile
    result: Any
    subject: str
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    channel_names: list[str]
    channel_index_by_norm: dict[str, int]
    time_axis_s: np.ndarray
    analysis_level: str
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]
    signature: SubjectStatsSignature


SignatureT = TypeVar("SignatureT", bound=SubjectStatsSignature)
InputT = TypeVar("InputT", bound=SubjectStatsInput)
ContributionT = TypeVar("ContributionT")


def hash_time_axis(time_axis_s: np.ndarray) -> str:
    """Return a stable content hash for a time axis."""

    return hashlib.sha1(np.asarray(time_axis_s, dtype=np.float64).tobytes()).hexdigest()


def stable_signature_json(value: Any) -> str:
    """Serialize a signature component into deterministic compact JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def normalize_extra_key_parts(
    extra_key_parts: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    """Return deterministic key/value pairs for pipeline-specific compatibility."""

    if not extra_key_parts:
        return ()
    return tuple(
        (str(key), stable_signature_json(extra_key_parts[key]))
        for key in sorted(extra_key_parts)
    )


def build_subject_stats_signature(
    stats_file: BIDSFile,
    result: Any,
    *,
    extra_key_parts: Mapping[str, Any] | None = None,
) -> SubjectStatsSignature:
    """Build a compatibility signature from common subject-level result fields."""

    time_axis_s = np.asarray(getattr(result, "time_axis_s", np.array([])), dtype=np.float64)
    metadata = getattr(result, "metadata", {}) or {}
    trial_activity_summary_source = getattr(
        result,
        "trial_activity_summary_source",
        metadata.get("trial_activity_summary_source", {}),
    )
    return SubjectStatsSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=(
            str(getattr(result, "condition_a", "condition_a")),
            str(getattr(result, "condition_b", "condition_b")),
        ),
        time_axis_hash=hash_time_axis(time_axis_s),
        time_axis_len=int(len(time_axis_s)),
        analysis_level=str(getattr(result, "analysis_level", "channel") or "channel"),
        binning_mode=str(metadata.get("binning_mode", "none") or "none"),
        window_ms=float(getattr(result, "window_ms", 0.0)),
        n_bins=int(getattr(result, "n_bins", 0)),
        effective_n_bins=int(metadata.get("effective_n_bins", len(time_axis_s))),
        activity_zscore=str(getattr(result, "activity_zscore", "none") or "none"),
        activity_baseline_tmin_s=float(
            getattr(result, "activity_baseline_tmin_s", -0.2)
        ),
        activity_baseline_tmax_s=float(
            getattr(result, "activity_baseline_tmax_s", 0.0)
        ),
        activity_baseline_scope=str(
            getattr(result, "activity_baseline_scope", "global") or "global"
        ),
        activity_baseline_remove_outlier_trial_means=bool(
            getattr(result, "activity_baseline_remove_outlier_trial_means", False)
        ),
        trial_activity_summary_kind=str(
            getattr(result, "trial_activity_summary_kind", "epoch_mean")
            or "epoch_mean"
        ),
        trial_activity_summary_missing_response_policy=str(
            getattr(
                result,
                "trial_activity_summary_missing_response_policy",
                "clamp_to_epoch",
            )
            or "clamp_to_epoch"
        ),
        trial_activity_summary_source_json=stable_signature_json(
            trial_activity_summary_source or {}
        ),
        trial_activity_summary_label=str(
            getattr(result, "trial_activity_summary_label", "Epoch mean activity")
            or "Epoch mean activity"
        ),
        extra_key_parts=normalize_extra_key_parts(extra_key_parts),
    )


def build_subject_stats_input(
    stats_file: BIDSFile,
    result: Any,
    *,
    extra_key_parts: Mapping[str, Any] | None = None,
) -> SubjectStatsInput:
    """Build the technical wrapper used by group-level processors."""

    raw_subject = str(stats_file.get("subject") or stats_file.get("sub") or "").strip()
    subject = normalize_subject_value(raw_subject)
    channel_names = list(getattr(result, "channel_names", []))
    channel_index_by_norm: dict[str, int] = {}
    for idx, name in enumerate(channel_names):
        key = normalize_channel_name(str(name))
        channel_index_by_norm.setdefault(key, idx)

    metadata = getattr(result, "metadata", {}) or {}
    time_axis_s = np.asarray(getattr(result, "time_axis_s", np.array([])), dtype=np.float64)
    return SubjectStatsInput(
        stats_file=stats_file,
        result=result,
        subject=subject,
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=(
            str(getattr(result, "condition_a", "condition_a")),
            str(getattr(result, "condition_b", "condition_b")),
        ),
        channel_names=channel_names,
        channel_index_by_norm=channel_index_by_norm,
        time_axis_s=time_axis_s,
        analysis_level=str(getattr(result, "analysis_level", "channel") or "channel"),
        binning_mode=str(metadata.get("binning_mode", "none") or "none"),
        window_ms=float(getattr(result, "window_ms", 0.0)),
        n_bins=int(getattr(result, "n_bins", 0)),
        effective_n_bins=int(metadata.get("effective_n_bins", len(time_axis_s))),
        source_ieeg_files=list(getattr(result, "source_ieeg_files", [])),
        source_electrodes_files=list(getattr(result, "source_electrodes_files", [])),
        signature=build_subject_stats_signature(
            stats_file,
            result,
            extra_key_parts=extra_key_parts,
        ),
    )


def load_result_via_public_loader(
    stats_file: BIDSFile,
    loader: Callable[[Path], Any],
) -> Any:
    """Load a result through its public loader, including attached HDF5 handles."""

    path = Path(stats_file.path)
    if path.exists():
        return loader(path)

    if getattr(stats_file, "is_loaded", False):
        import h5py

        with stats_file.ensure_loaded() as loaded:
            if isinstance(loaded, h5py.File):
                fd, tmp_name = tempfile.mkstemp(suffix=path.suffix or ".h5")
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    with h5py.File(tmp_path, "w") as out:
                        for key in loaded.keys():
                            loaded.copy(key, out)
                    return loader(tmp_path)
                finally:
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass

    return loader(path)


def build_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    read_signature: Callable[[BIDSFile], SignatureT],
) -> list[BIDSFileGroup]:
    """Group subject-level stats files by processing compatibility."""

    if not stats_files:
        return []

    groups: dict[tuple[Any, ...], list[BIDSFile]] = {}
    for file in sorted(stats_files, key=lambda item: str(item.path)):
        signature = read_signature(file)
        groups.setdefault(signature.key, []).append(file)

    out_groups: list[BIDSFileGroup] = []
    for key in sorted(groups, key=str):
        files = groups[key]
        out_groups.append(BIDSFileGroup(primary=files[0], secondaries=files[1:]))
    return out_groups


def validate_group_compatibility(
    inputs: Sequence[InputT],
    *,
    empty_message: str,
    non_channel_message: str,
    incompatible_message: str,
) -> None:
    """Validate that a set of loaded subject inputs can be processed together."""

    if not inputs:
        raise ValueError(empty_message)
    if any(item.analysis_level != "channel" for item in inputs):
        non_channel = [
            item.stats_file.path.name
            for item in inputs
            if item.analysis_level != "channel"
        ]
        raise ValueError(f"{non_channel_message} Found non-channel files: {', '.join(non_channel)}.")

    reference = inputs[0].signature.key
    for item in inputs[1:]:
        if item.signature.key != reference:
            raise ValueError(incompatible_message)
