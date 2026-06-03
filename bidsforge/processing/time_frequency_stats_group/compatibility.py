from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.utils.channels import normalize_channel_name


@dataclass(frozen=True)
class TFSubjectStatsSignature:
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    frequency_hash: str
    time_hash: str
    n_freqs: int
    n_times: int
    power_mode: str
    time_selection: str
    baseline_grand_average: bool
    extra_key_parts: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.source_desc,
            self.condition_labels,
            self.frequency_hash,
            self.time_hash,
            self.n_freqs,
            self.n_times,
            self.power_mode,
            self.time_selection,
            self.baseline_grand_average,
            self.extra_key_parts,
        )


@dataclass(frozen=True)
class TFSubjectStatsInput:
    stats_file: BIDSFile
    result: Any
    subject: str
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    channel_names: list[str]
    channel_index_by_norm: dict[str, int]
    frequency_hz: np.ndarray
    time_axis_s: np.ndarray
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]
    signature: TFSubjectStatsSignature


InputT = TypeVar("InputT", bound=TFSubjectStatsInput)
SignatureT = TypeVar("SignatureT", bound=TFSubjectStatsSignature)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_axis(values: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def normalize_extra_key_parts(extra_key_parts: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not extra_key_parts:
        return ()
    return tuple((str(key), stable_json(extra_key_parts[key])) for key in sorted(extra_key_parts))


def build_tf_subject_signature(
    stats_file: BIDSFile,
    result: Any,
    *,
    extra_key_parts: Mapping[str, Any] | None = None,
) -> TFSubjectStatsSignature:
    freqs = np.asarray(getattr(result, "frequency_hz", np.array([])), dtype=np.float64)
    times = np.asarray(getattr(result, "time_axis_s", np.array([])), dtype=np.float64)
    meta = getattr(result, "metadata", {}) or {}
    return TFSubjectStatsSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=(
            str(getattr(result, "condition_a", "condition_a")),
            str(getattr(result, "condition_b", "condition_b")),
        ),
        frequency_hash=hash_axis(freqs),
        time_hash=hash_axis(times),
        n_freqs=int(freqs.size),
        n_times=int(times.size),
        power_mode=str(getattr(result, "power_mode", meta.get("power_mode", "stored"))),
        time_selection=str(meta.get("time_selection", "")),
        baseline_grand_average=bool(meta.get("baseline_grand_average", False)),
        extra_key_parts=normalize_extra_key_parts(extra_key_parts),
    )


def build_tf_subject_input(
    stats_file: BIDSFile,
    result: Any,
    *,
    extra_key_parts: Mapping[str, Any] | None = None,
) -> TFSubjectStatsInput:
    subject = normalize_subject_value(str(stats_file.get("subject") or stats_file.get("sub") or ""))
    channels = list(getattr(result, "channel_names", []))
    return TFSubjectStatsInput(
        stats_file=stats_file,
        result=result,
        subject=subject,
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=(
            str(getattr(result, "condition_a", "condition_a")),
            str(getattr(result, "condition_b", "condition_b")),
        ),
        channel_names=channels,
        channel_index_by_norm={
            normalize_channel_name(str(channel)): idx for idx, channel in enumerate(channels)
        },
        frequency_hz=np.asarray(getattr(result, "frequency_hz", np.array([])), dtype=np.float64),
        time_axis_s=np.asarray(getattr(result, "time_axis_s", np.array([])), dtype=np.float64),
        source_ieeg_files=list(getattr(result, "source_ieeg_files", [])),
        source_electrodes_files=list(getattr(result, "source_electrodes_files", [])),
        signature=build_tf_subject_signature(
            stats_file,
            result,
            extra_key_parts=extra_key_parts,
        ),
    )


def load_result_via_public_loader(stats_file: BIDSFile, loader: Callable[[Path], Any]) -> Any:
    return loader(Path(stats_file.path))


def build_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    read_signature: Callable[[BIDSFile], SignatureT],
) -> list[BIDSFileGroup]:
    groups: dict[tuple[Any, ...], list[BIDSFile]] = {}
    for stats_file in sorted(stats_files, key=lambda item: str(item.path)):
        sig = read_signature(stats_file)
        groups.setdefault(sig.key, []).append(stats_file)
    return [
        BIDSFileGroup(primary=files[0], secondaries=files[1:])
        for _, files in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def validate_group_compatibility(inputs: Sequence[InputT]) -> None:
    if not inputs:
        raise ValueError("At least one TF subject stats input is required.")
    reference = inputs[0].signature.key
    for item in inputs[1:]:
        if item.signature.key != reference:
            raise ValueError(
                "Incompatible time_frequency_stats inputs. "
                "Use the build_*_compatible_groups helper to split heterogeneous files."
            )
