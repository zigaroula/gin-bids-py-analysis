"""
Inspect trial-stats outputs (.h5 + optional companion _trials.tsv) and generate a
visual summary. Supports subject-level `condition_test` / `regression` outputs
and group-level `condition_test_group` / `regression_group` outputs.

Examples:
  python scripts/inspect_trial_stats.py --input E:\\CBT\\bids\\derivatives\\condition_test\\sub-01\\ieeg\\sub-01_desc-conditiontest_stats.h5
  python scripts/inspect_trial_stats.py --bids-root E:\\CBT\\bids
  python scripts/inspect_trial_stats.py --bids-root E:\\CBT\\bids --no-save-figure
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import Iterable

import h5py
import numpy as np


DEFAULT_BIDS_ROOT = Path(r"D:\data_clarissa\valuation\bids")
DEFAULT_STATS_GLOB = "derivatives/**/*_stats.h5"

# ---------------------------------------------------------------------------
# Display options — toggle these to control which figure panels are rendered.
# ---------------------------------------------------------------------------
SHOW_HEATMAP = True        # difference heatmap (multi-bin mode)
SHOW_BAR_CHART = True      # condition-means bar chart (single-bin mode)
SHOW_TABLE = False         # per-channel/region statistics table
SHOW_PARAMS_PANEL = False  # metadata text box (right side of figure)
HEATMAP_ROW_HEIGHT_PX = 20  # pixel height per channel/region row in the heatmap
HEATMAP_METRIC = "t_value"  # heatmap data: "difference" (mean_difference) or "t_value"


@dataclass
class TrialStatsSnapshot:
    stats_path: Path
    channels: list[str]
    time_s: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    p_values_uncorrected: np.ndarray
    significant_mask: np.ndarray
    condition_a_mean: np.ndarray
    condition_b_mean: np.ndarray
    condition_a_sem: np.ndarray
    condition_b_sem: np.ndarray
    mean_difference: np.ndarray
    difference_sem: np.ndarray
    difference_ci95_low: np.ndarray
    difference_ci95_high: np.ndarray
    condition_labels: tuple[str, str]
    trial_counts: tuple[int, int]
    stats_valid: bool
    correction_method: str
    significance_alpha: float
    analysis_level: str
    atlas_name: str
    atlas_regions: list[str]
    region_channels: dict[str, list[str]]
    window_ms: float
    n_bins: int
    effective_n_bins: int
    binning_mode: str
    source_metric: str = ""
    roi_mode: str = ""
    pipeline_name: str = "conditiontest"
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass(frozen=True)
class RegionMetric:
    label: str
    mean_a_global: float
    mean_b_global: float
    delta_mean_global: float
    p_value: float
    p_raw: float
    t_value: float
    sem_diff: float
    ci95_low: float
    ci95_high: float
    significant: bool
    p_min: float
    p_raw_min: float
    t_at_p_min: float
    sem_diff_at_p_min: float
    ci95_low_at_p_min: float
    ci95_high_at_p_min: float
    sig_bins: int
    sig_any: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect trial-stats outputs and render a summary figure.",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=DEFAULT_BIDS_ROOT,
        help="BIDS root used with --glob (default: %(default)s)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=DEFAULT_STATS_GLOB,
        help=(
            "Glob relative to --bids-root when --input is not provided "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="One or many explicit *_stats.h5 files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top channels used in the compact console recap.",
    )
    parser.add_argument(
        "--save-channel-summary",
        action="store_true",
        help="Write a <stem>_channel_summary.tsv next to each stats file.",
    )
    parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a <stem>_summary.png next to each stats file (default: enabled).",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=160,
        help="DPI used when saving PNG figures (default: %(default)s).",
    )
    return parser.parse_args()


def _decode_str_array(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _bool_scalar(dataset: h5py.Dataset | None, default: bool = False) -> bool:
    if dataset is None:
        return default
    try:
        return bool(dataset[()])
    except Exception:
        return default


def _float_scalar(dataset: h5py.Dataset | None, default: float = np.nan) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[()])
    except Exception:
        return default


def _str_scalar(dataset: h5py.Dataset | None, default: str = "") -> str:
    if dataset is None:
        return default
    try:
        return str(dataset.asstr()[()])
    except Exception:
        return default


def _load_optional_float_dataset(group: h5py.Group, name: str) -> np.ndarray | None:
    dataset = group.get(name)
    if dataset is None:
        return None
    try:
        if dataset.shape == ():
            return np.asarray([float(dataset[()])], dtype=np.float64)
        return np.asarray(dataset[:], dtype=np.float64)
    except Exception:
        return None


def _coerce_float_feature_time(
    values: np.ndarray | None,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    if values is None:
        return out

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape == (n_features, n_times):
            return arr
        if arr.shape == (n_times, n_features):
            return arr.T
        rows = min(n_features, arr.shape[0])
        cols = min(n_times, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
        return out

    if arr.ndim == 1:
        if n_times == 1 and arr.size == n_features:
            return arr.reshape(n_features, 1)
        if n_features == 1 and arr.size == n_times:
            return arr.reshape(1, n_times)
        if n_features > 0 and n_times > 0 and arr.size == n_features * n_times:
            return arr.reshape(n_features, n_times)
        flat = out.reshape(-1)
        limit = min(flat.size, arr.size)
        flat[:limit] = arr[:limit]
        return out

    return out


def _coerce_bool_feature_time(
    values: np.ndarray | None,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    out = np.zeros((n_features, n_times), dtype=bool)
    if values is None:
        return out

    arr = np.asarray(values, dtype=bool)
    if arr.ndim == 2:
        if arr.shape == (n_features, n_times):
            return arr
        if arr.shape == (n_times, n_features):
            return arr.T
        rows = min(n_features, arr.shape[0])
        cols = min(n_times, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
        return out

    if arr.ndim == 1:
        if n_times == 1 and arr.size == n_features:
            return arr.reshape(n_features, 1)
        if n_features == 1 and arr.size == n_times:
            return arr.reshape(1, n_times)
        if n_features > 0 and n_times > 0 and arr.size == n_features * n_times:
            return arr.reshape(n_features, n_times)
        flat = out.reshape(-1)
        limit = min(flat.size, arr.size)
        flat[:limit] = arr[:limit]
        return out

    return out


def _is_region_level(analysis_level: str) -> bool:
    return analysis_level in {"roi", "roi_group"}


def _load_group_region_channels(
    fh: h5py.File,
    channels: list[str],
) -> dict[str, list[str]]:
    region_channels: dict[str, list[str]] = {region: [] for region in channels}
    if "contributions" not in fh:
        return region_channels

    contrib_grp = fh["contributions"]
    if (
        "region" not in contrib_grp
        or "channel" not in contrib_grp
        or "subject" not in contrib_grp
    ):
        return region_channels

    regions = _decode_str_array(np.asarray(contrib_grp["region"][:]))
    channel_names = _decode_str_array(np.asarray(contrib_grp["channel"][:]))
    subjects = _decode_str_array(np.asarray(contrib_grp["subject"][:]))
    if not (len(regions) == len(channel_names) == len(subjects)):
        return region_channels

    for region, subject, channel in zip(regions, subjects, channel_names):
        key = f"sub-{subject}:{channel}"
        region_list = region_channels.setdefault(region, [])
        if key not in region_list:
            region_list.append(key)

    return region_channels


def _load_snapshot(stats_path: Path) -> TrialStatsSnapshot:
    with h5py.File(stats_path, "r") as fh:
        stats_grp = fh["stats"]
        axes_grp = fh["axes"]
        meta_grp = fh["meta"]
        means_grp = fh["means"]
        uncertainty_grp = fh["uncertainty"]

        analysis_level = _str_scalar(meta_grp.get("analysis_level"), default="channel")
        axis_name = "region" if _is_region_level(analysis_level) else "channel"
        channels = _decode_str_array(np.asarray(axes_grp[axis_name][:]))
        time_s = np.asarray(axes_grp["time_s"][:], dtype=np.float64)
        n_features = len(channels)
        n_times = len(time_s)

        t_values = _coerce_float_feature_time(
            np.asarray(stats_grp["t_values"][:], dtype=np.float64),
            n_features,
            n_times,
        )
        p_values = _coerce_float_feature_time(
            np.asarray(stats_grp["p_values"][:], dtype=np.float64),
            n_features,
            n_times,
        )
        p_values_uncorrected = _coerce_float_feature_time(
            np.asarray(stats_grp["p_values_uncorrected"][:], dtype=np.float64)
            if "p_values_uncorrected" in stats_grp
            else p_values.copy(),
            n_features,
            n_times,
        )

        significance_alpha = _float_scalar(meta_grp.get("significance_alpha"), default=0.05)
        correction_method = _str_scalar(
            meta_grp.get("p_value_correction_method"),
            default="unknown",
        )
        if "significant_mask" in stats_grp:
            significant_mask = _coerce_bool_feature_time(
                np.asarray(stats_grp["significant_mask"][:], dtype=bool),
                n_features,
                n_times,
            )
        else:
            significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

        prov_grp = fh["provenance"] if "provenance" in fh else None
        pipeline_name = (
            _str_scalar(prov_grp.get("pipeline_name"), default="conditiontest")
            if prov_grp is not None
            else "conditiontest"
        )
        source_metric = _str_scalar(meta_grp.get("source_metric"), default="")
        roi_mode = _str_scalar(meta_grp.get("roi_mode"), default="")
        is_group_level = analysis_level == "roi_group" or (
            "metric_mean" in means_grp and "difference" not in means_grp
        )

        if is_group_level:
            if not source_metric:
                source_metric = "metric_mean"
            condition_labels = (source_metric, "zero")
            metric_mean_raw = _load_optional_float_dataset(means_grp, "metric_mean")
            condition_a_mean = _coerce_float_feature_time(metric_mean_raw, n_features, n_times)
            condition_b_mean = np.zeros((n_features, n_times), dtype=np.float64)
            mean_difference = condition_a_mean.copy()

            metric_sem_raw = _load_optional_float_dataset(uncertainty_grp, "metric_sem")
            difference_sem = _coerce_float_feature_time(metric_sem_raw, n_features, n_times)
            condition_a_sem = difference_sem.copy()
            condition_b_sem = np.zeros((n_features, n_times), dtype=np.float64)
            difference_ci95_low = condition_a_mean - (1.96 * difference_sem)
            difference_ci95_high = condition_a_mean + (1.96 * difference_sem)
            trial_counts = (0, 0)
            stats_valid = bool(n_features > 0)
        else:
            if "trial_count_labels" in meta_grp:
                raw_labels = _decode_str_array(np.asarray(meta_grp["trial_count_labels"][:]))
                if len(raw_labels) >= 2:
                    condition_labels = (raw_labels[0], raw_labels[1])
                else:
                    condition_labels = ("condition_a", "condition_b")
            else:
                non_difference_keys = [name for name in means_grp.keys() if name != "difference"]
                if len(non_difference_keys) >= 2:
                    condition_labels = (non_difference_keys[0], non_difference_keys[1])
                else:
                    condition_labels = ("condition_a", "condition_b")

            mean_a_raw = _load_optional_float_dataset(means_grp, condition_labels[0])
            mean_b_raw = _load_optional_float_dataset(means_grp, condition_labels[1])
            if mean_a_raw is None:
                mean_a_raw = _load_optional_float_dataset(means_grp, "condition_a")
            if mean_b_raw is None:
                mean_b_raw = _load_optional_float_dataset(means_grp, "condition_b")

            non_difference_keys = [name for name in means_grp.keys() if name != "difference"]
            if mean_a_raw is None and non_difference_keys:
                mean_a_raw = _load_optional_float_dataset(means_grp, non_difference_keys[0])
                condition_labels = (non_difference_keys[0], condition_labels[1])
            if mean_b_raw is None:
                for key in non_difference_keys:
                    if key == condition_labels[0]:
                        continue
                    candidate = _load_optional_float_dataset(means_grp, key)
                    if candidate is not None:
                        mean_b_raw = candidate
                        condition_labels = (condition_labels[0], key)
                        break

            condition_a_mean = _coerce_float_feature_time(mean_a_raw, n_features, n_times)
            condition_b_mean = _coerce_float_feature_time(mean_b_raw, n_features, n_times)

            mean_difference_raw = _load_optional_float_dataset(means_grp, "difference")
            if mean_difference_raw is None:
                mean_difference_raw = condition_a_mean - condition_b_mean
            mean_difference = _coerce_float_feature_time(mean_difference_raw, n_features, n_times)

            cond_a_sem_raw = _load_optional_float_dataset(
                uncertainty_grp, condition_labels[0] + "_sem"
            )
            if cond_a_sem_raw is None:
                cond_a_sem_raw = _load_optional_float_dataset(uncertainty_grp, "condition_a_sem")
            cond_b_sem_raw = _load_optional_float_dataset(
                uncertainty_grp, condition_labels[1] + "_sem"
            )
            if cond_b_sem_raw is None:
                cond_b_sem_raw = _load_optional_float_dataset(uncertainty_grp, "condition_b_sem")
            condition_a_sem = _coerce_float_feature_time(cond_a_sem_raw, n_features, n_times)
            condition_b_sem = _coerce_float_feature_time(cond_b_sem_raw, n_features, n_times)

            difference_sem_raw = _load_optional_float_dataset(uncertainty_grp, "difference_sem")
            if difference_sem_raw is None:
                difference_sem_raw = np.sqrt(
                    np.square(condition_a_sem, dtype=np.float64)
                    + np.square(condition_b_sem, dtype=np.float64)
                )
            difference_sem = _coerce_float_feature_time(
                difference_sem_raw, n_features, n_times
            )

            difference_ci95_low_raw = _load_optional_float_dataset(
                uncertainty_grp, "difference_ci95_low"
            )
            difference_ci95_high_raw = _load_optional_float_dataset(
                uncertainty_grp, "difference_ci95_high"
            )
            if difference_ci95_low_raw is None or difference_ci95_high_raw is None:
                difference_ci95_low = mean_difference - (1.96 * difference_sem)
                difference_ci95_high = mean_difference + (1.96 * difference_sem)
            else:
                difference_ci95_low = _coerce_float_feature_time(
                    difference_ci95_low_raw, n_features, n_times
                )
                difference_ci95_high = _coerce_float_feature_time(
                    difference_ci95_high_raw, n_features, n_times
                )

            if "trial_counts" in meta_grp:
                raw_counts = np.asarray(meta_grp["trial_counts"][:], dtype=np.int64)
                if raw_counts.size >= 2:
                    trial_counts = (int(raw_counts[0]), int(raw_counts[1]))
                else:
                    trial_counts = (0, 0)
            else:
                trial_counts = (0, 0)

            stats_valid = _bool_scalar(meta_grp.get("stats_valid"), default=False)

        atlas_name = _str_scalar(meta_grp.get("atlas_name"), default="")
        if is_group_level:
            atlas_regions = channels if _is_region_level(analysis_level) else []
            region_channels = _load_group_region_channels(fh, channels)
        else:
            atlas_regions = (
                _decode_str_array(np.asarray(meta_grp["atlas_regions"][:]))
                if "atlas_regions" in meta_grp
                else []
            )
            region_channels = {}
            if "atlas_region_channel_map" in meta_grp:
                atlas_map_grp = meta_grp["atlas_region_channel_map"]
                map_region_order = _decode_str_array(
                    np.asarray(atlas_map_grp["region_order"][:], dtype=object)
                ) if "region_order" in atlas_map_grp else []
                map_regions = _decode_str_array(
                    np.asarray(atlas_map_grp["region"][:], dtype=object)
                ) if "region" in atlas_map_grp else []
                map_channels = _decode_str_array(
                    np.asarray(atlas_map_grp["channel"][:], dtype=object)
                ) if "channel" in atlas_map_grp else []
                if len(map_regions) != len(map_channels):
                    raise ValueError(
                        "meta/atlas_region_channel_map/region and /channel must have "
                        f"the same length in {stats_path.name}."
                    )
                region_channels = {region: [] for region in map_region_order}
                for region, channel in zip(map_regions, map_channels):
                    region_list = region_channels.setdefault(region, [])
                    if channel not in region_list:
                        region_list.append(channel)
            if _is_region_level(analysis_level):
                for region in channels:
                    region_channels.setdefault(region, [])

        summary_grp = fh["summary_epoch"] if "summary_epoch" in fh else None
        if "roi_channel_counts" in meta_grp:
            roi_channel_counts = np.asarray(
                meta_grp["roi_channel_counts"][:], dtype=np.int64
            )
        elif summary_grp is not None and "roi_channel_counts" in summary_grp:
            roi_channel_counts = np.asarray(
                summary_grp["roi_channel_counts"][:], dtype=np.int64
            )
        else:
            roi_channel_counts = np.asarray([], dtype=np.int64)
        if "roi_subject_counts" in meta_grp:
            roi_subject_counts = np.asarray(
                meta_grp["roi_subject_counts"][:], dtype=np.int64
            )
        elif summary_grp is not None and "roi_subject_counts" in summary_grp:
            roi_subject_counts = np.asarray(
                summary_grp["roi_subject_counts"][:], dtype=np.int64
            )
        else:
            roi_subject_counts = np.asarray([], dtype=np.int64)

        if "window_ms" in meta_grp:
            window_ms = _float_scalar(meta_grp.get("window_ms"), default=0.0)
        else:
            # Backward compatibility with older trial-stats outputs.
            window_ms = _float_scalar(meta_grp.get("temporal_window_ms"), default=0.0)
        if "n_bins" in meta_grp:
            n_bins = int(np.asarray(meta_grp["n_bins"][()], dtype=np.int64))
        else:
            n_bins = 0
        if "effective_n_bins" in meta_grp:
            effective_n_bins = int(np.asarray(meta_grp["effective_n_bins"][()], dtype=np.int64))
        else:
            effective_n_bins = int(len(time_s))
        binning_mode = _str_scalar(meta_grp.get("binning_mode"), default="")
        if not binning_mode:
            if window_ms > 0:
                binning_mode = "window_ms"
            elif n_bins > 0:
                binning_mode = "n_bins"
            else:
                binning_mode = "none"

    return TrialStatsSnapshot(
        stats_path=stats_path,
        channels=channels,
        time_s=time_s,
        t_values=t_values,
        p_values=p_values,
        p_values_uncorrected=p_values_uncorrected,
        significant_mask=significant_mask,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_sem=condition_a_sem,
        condition_b_sem=condition_b_sem,
        mean_difference=mean_difference,
        difference_sem=difference_sem,
        difference_ci95_low=difference_ci95_low,
        difference_ci95_high=difference_ci95_high,
        condition_labels=condition_labels,
        trial_counts=trial_counts,
        stats_valid=stats_valid,
        correction_method=correction_method,
        significance_alpha=significance_alpha,
        analysis_level=analysis_level,
        atlas_name=atlas_name,
        atlas_regions=atlas_regions,
        region_channels=region_channels,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        binning_mode=binning_mode,
        primary_condition_metric=source_metric,
        roi_mode=roi_mode,
        pipeline_name=pipeline_name,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
    )


def _trial_table_path(stats_path: Path) -> Path:
    tsv_path = stats_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


def _load_trial_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader]


def _summarize_exclusions(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        keep = str(row.get("keep", "")).strip().lower() == "true"
        if keep:
            continue
        reason = (row.get("exclusion_reason") or "unspecified").strip() or "unspecified"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _safe_row_nanmin(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        row = values[idx]
        finite = np.isfinite(row)
        if finite.any():
            out[idx] = float(np.nanmin(row))
    return out


def _safe_row_nanmax(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        row = values[idx]
        finite = np.isfinite(row)
        if finite.any():
            out[idx] = float(np.nanmax(row))
    return out


def _safe_row_nanmean(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        row = values[idx]
        finite = np.isfinite(row)
        if finite.any():
            out[idx] = float(np.nanmean(row))
    return out


def _rank_channels(snapshot: TrialStatsSnapshot) -> list[dict[str, float | str]]:
    n_times = snapshot.p_values.shape[1] if snapshot.p_values.ndim == 2 else 0
    if n_times == 0:
        return []

    sig_counts = snapshot.significant_mask.sum(axis=1)
    sig_fraction = sig_counts.astype(np.float64) / float(n_times)

    min_p = _safe_row_nanmin(snapshot.p_values)
    min_p_uncorrected = _safe_row_nanmin(snapshot.p_values_uncorrected)
    max_abs_t = _safe_row_nanmax(np.abs(snapshot.t_values))
    max_abs_diff = _safe_row_nanmax(np.abs(snapshot.mean_difference))

    ranking: list[dict[str, float | str]] = []
    for idx, channel in enumerate(snapshot.channels):
        ranking.append(
            {
                "channel": channel,
                "sig_fraction": float(sig_fraction[idx]),
                "min_p": float(min_p[idx]),
                "min_p_uncorrected": float(min_p_uncorrected[idx]),
                "max_abs_t": float(max_abs_t[idx]),
                "max_abs_diff": float(max_abs_diff[idx]),
            }
        )

    ranking.sort(
        key=lambda row: (
            float(row["sig_fraction"]),
            float(row["max_abs_t"]),
            -float(row["min_p"]),
        ),
        reverse=True,
    )
    return ranking


def _write_channel_summary_tsv(
    stats_path: Path,
    rows: list[dict[str, float | str]],
) -> Path:
    out_path = stats_path.with_name(stats_path.stem.replace("_stats", "") + "_channel_summary.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "channel",
                "sig_fraction",
                "min_p",
                "min_p_uncorrected",
                "max_abs_t",
                "max_abs_diff",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _prepare_region_metrics(
    snapshot: TrialStatsSnapshot,
) -> tuple[list[RegionMetric], bool]:
    n_regions = len(snapshot.channels)
    n_times = snapshot.p_values.shape[1] if snapshot.p_values.ndim == 2 else 0
    has_multi_bin = n_times > 1 and snapshot.effective_n_bins > 1

    mean_a_global = _safe_row_nanmean(snapshot.condition_a_mean)
    mean_b_global = _safe_row_nanmean(snapshot.condition_b_mean)
    sig_counts = (
        snapshot.significant_mask.sum(axis=1).astype(np.int64)
        if snapshot.significant_mask.ndim == 2
        else np.zeros(n_regions, dtype=np.int64)
    )

    rows: list[RegionMetric] = []
    for idx, label in enumerate(snapshot.channels):
        delta_global = float(mean_a_global[idx] - mean_b_global[idx])
        if has_multi_bin:
            p_row = snapshot.p_values[idx] if n_times else np.empty((0,), dtype=np.float64)
            p_raw_row = (
                snapshot.p_values_uncorrected[idx]
                if n_times
                else np.empty((0,), dtype=np.float64)
            )
            finite_idx = np.flatnonzero(np.isfinite(p_row))
            if finite_idx.size:
                argmin_local = int(np.argmin(p_row[finite_idx]))
                best_idx = int(finite_idx[argmin_local])
                p_min = float(p_row[best_idx])
                t_at_p_min = float(snapshot.t_values[idx, best_idx])
                p_raw_at_p_min = float(p_raw_row[best_idx]) if best_idx < p_raw_row.size else np.nan
                sem_at_p_min = float(snapshot.difference_sem[idx, best_idx])
                ci_low_at_p_min = float(snapshot.difference_ci95_low[idx, best_idx])
                ci_high_at_p_min = float(snapshot.difference_ci95_high[idx, best_idx])
            else:
                p_min = np.nan
                t_at_p_min = np.nan
                p_raw_at_p_min = np.nan
                sem_at_p_min = np.nan
                ci_low_at_p_min = np.nan
                ci_high_at_p_min = np.nan

            finite_raw_idx = np.flatnonzero(np.isfinite(p_raw_row))
            if finite_raw_idx.size:
                p_raw_min = float(np.nanmin(p_raw_row[finite_raw_idx]))
            else:
                p_raw_min = np.nan

            rows.append(
                RegionMetric(
                    label=label,
                    mean_a_global=float(mean_a_global[idx]),
                    mean_b_global=float(mean_b_global[idx]),
                    delta_mean_global=delta_global,
                    p_value=np.nan,
                    p_raw=np.nan,
                    t_value=np.nan,
                    sem_diff=np.nan,
                    ci95_low=np.nan,
                    ci95_high=np.nan,
                    significant=False,
                    p_min=p_min,
                    p_raw_min=p_raw_min if np.isfinite(p_raw_min) else p_raw_at_p_min,
                    t_at_p_min=t_at_p_min,
                    sem_diff_at_p_min=sem_at_p_min,
                    ci95_low_at_p_min=ci_low_at_p_min,
                    ci95_high_at_p_min=ci_high_at_p_min,
                    sig_bins=int(sig_counts[idx]),
                    sig_any=bool(sig_counts[idx] > 0),
                )
            )
        else:
            p_value = float(snapshot.p_values[idx, 0]) if n_times else np.nan
            p_raw = float(snapshot.p_values_uncorrected[idx, 0]) if n_times else np.nan
            t_value = float(snapshot.t_values[idx, 0]) if n_times else np.nan
            sem_diff = float(snapshot.difference_sem[idx, 0]) if n_times else np.nan
            ci95_low = float(snapshot.difference_ci95_low[idx, 0]) if n_times else np.nan
            ci95_high = float(snapshot.difference_ci95_high[idx, 0]) if n_times else np.nan
            significant = bool(snapshot.significant_mask[idx, 0]) if n_times else False
            rows.append(
                RegionMetric(
                    label=label,
                    mean_a_global=float(mean_a_global[idx]),
                    mean_b_global=float(mean_b_global[idx]),
                    delta_mean_global=delta_global,
                    p_value=p_value,
                    p_raw=p_raw,
                    t_value=t_value,
                    sem_diff=sem_diff,
                    ci95_low=ci95_low,
                    ci95_high=ci95_high,
                    significant=significant,
                    p_min=np.nan,
                    p_raw_min=np.nan,
                    t_at_p_min=np.nan,
                    sem_diff_at_p_min=np.nan,
                    ci95_low_at_p_min=np.nan,
                    ci95_high_at_p_min=np.nan,
                    sig_bins=int(sig_counts[idx]),
                    sig_any=bool(sig_counts[idx] > 0),
                )
            )
    return rows, has_multi_bin


def _format_float(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}g}"


def _item_label(snapshot: TrialStatsSnapshot) -> str:
    return "Region" if _is_region_level(snapshot.analysis_level) else "Channel"


def _region_channel_panel_lines(snapshot: TrialStatsSnapshot) -> list[str]:
    if not _is_region_level(snapshot.analysis_level):
        return ["region_channels: n/a (analysis_level=channel)"]

    ordered_regions = snapshot.channels if snapshot.channels else list(snapshot.region_channels.keys())
    if not ordered_regions:
        return ["region_channels: n/a"]

    lines = ["region_channels:"]
    for region in ordered_regions:
        channels = snapshot.region_channels.get(region, [])
        if not channels:
            lines.append(f"  {region}: -")
            continue

        payload = ", ".join(channels)
        indent = " " * (len(region) + 4)
        wrapper = textwrap.TextWrapper(
            width=70,
            initial_indent=f"  {region}: ",
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped = wrapper.wrap(payload)
        lines.extend(wrapped if wrapped else [f"  {region}: -"])
    return lines


def _figure_output_path(stats_path: Path) -> Path:
    return stats_path.with_name(f"{stats_path.stem}_summary.png")


def _load_pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate summary figures. "
            "Install it with: pip install matplotlib"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_summary_figure(
    snapshot: TrialStatsSnapshot,
    *,
    dpi: int,
    show_heatmap: bool = True,
    show_bar_chart: bool = True,
    show_table: bool = True,
    show_params_panel: bool = True,
    heatmap_metric: str = "difference",
) -> Path | None:
    plt = _load_pyplot()

    metrics, has_multi_bin = _prepare_region_metrics(snapshot)
    item_label = _item_label(snapshot)
    cond_a, cond_b = snapshot.condition_labels
    n_a, n_b = snapshot.trial_counts
    n_regions = max(len(metrics), 1)

    bar_a = np.array([row.mean_a_global for row in metrics], dtype=np.float64)
    bar_b = np.array([row.mean_b_global for row in metrics], dtype=np.float64)
    region_labels = [row.label for row in metrics]

    fig_width = float(np.clip(0.55 * n_regions + 6.5, 10.0, 30.0))
    table_height = float(np.clip(0.18 * n_regions + 1.4, 2.2, 8.8))

    top_panel_wanted = (has_multi_bin and show_heatmap) or (not has_multi_bin and show_bar_chart)
    if not top_panel_wanted and not show_table:
        return None

    # For the heatmap panel, derive height from a fixed per-row pixel budget so
    # every channel row has the same apparent height regardless of channel count.
    heatmap_panel_h_in = max(1.5, n_regions * HEATMAP_ROW_HEIGHT_PX / dpi)

    panel_names: list[str] = []
    height_ratios: list[float] = []
    if top_panel_wanted:
        panel_names.append("top")
        height_ratios.append(heatmap_panel_h_in if has_multi_bin else 3.5)
    if show_table:
        panel_names.append("table")
        height_ratios.append(
            max(2.9, table_height * 1.05) if has_multi_bin else max(1.8, table_height / 2.0)
        )

    # Build total figure height so each panel gets approximately its requested
    # height. The subplot area is ~85 % of the figure (top=0.92, bottom=0.07).
    figure_height = max(3.5, sum(height_ratios) / 0.85)

    fig, raw_axes = plt.subplots(
        len(panel_names),
        1,
        figsize=(fig_width, figure_height),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    flat_axes = raw_axes.ravel()

    ax_idx = 0
    ax_heat = None
    ax_bar = None
    ax_table = None
    if top_panel_wanted:
        if has_multi_bin:
            ax_heat = flat_axes[ax_idx]
        else:
            ax_bar = flat_axes[ax_idx]
        ax_idx += 1
    if show_table:
        ax_table = flat_axes[ax_idx]

    if ax_bar is not None:
        if metrics:
            x = np.arange(len(metrics), dtype=np.float64)
            bar_width = 0.42
            ax_bar.bar(
                x - (bar_width / 2.0),
                bar_a,
                width=bar_width,
                label=cond_a,
                color="#1f77b4",
                alpha=0.9,
            )
            ax_bar.bar(
                x + (bar_width / 2.0),
                bar_b,
                width=bar_width,
                label=cond_b,
                color="#ff7f0e",
                alpha=0.9,
            )
            ax_bar.set_xticks(x)
            tick_font_size = 9 if len(metrics) <= 20 else 7
            ax_bar.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=tick_font_size)
        else:
            ax_bar.text(
                0.5,
                0.5,
                f"No {item_label.lower()} data available.",
                ha="center",
                va="center",
                transform=ax_bar.transAxes,
            )
            ax_bar.set_xticks([])

        ax_bar.set_ylabel("Mean signal")
        ax_bar.set_title(f"{item_label}-level {cond_a} vs {cond_b}")
        ax_bar.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax_bar.legend(loc="upper left")

    if has_multi_bin and ax_heat is not None:
        if heatmap_metric == "t_value":
            heat_data = snapshot.t_values
            colorbar_label = f"t-value ({cond_a} vs {cond_b})"
        else:
            heat_data = snapshot.mean_difference
            colorbar_label = f"{cond_a} - {cond_b}"
        finite = heat_data[np.isfinite(heat_data)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if vmax == 0.0:
            vmax = 1.0

        heat = ax_heat.imshow(
            heat_data,
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        colorbar = fig.colorbar(heat, ax=ax_heat, fraction=0.03, pad=0.01)
        colorbar.set_label(colorbar_label, fontsize=9, labelpad=2)

        n_time_points = heat_data.shape[1]
        if n_time_points > 0:
            tick_count = min(8, n_time_points)
            tick_indices = np.unique(
                np.linspace(0, n_time_points - 1, num=tick_count, dtype=int)
            )
            ax_heat.set_xticks(tick_indices)
            ax_heat.set_xticklabels([f"{snapshot.time_s[idx]:.2f}" for idx in tick_indices])
        ax_heat.set_xlabel("Time (s)")

        region_tick_indices = np.arange(len(region_labels), dtype=int)
        # Scale font so labels fit within the fixed per-row pixel budget.
        ytick_fontsize = max(3, min(7, HEATMAP_ROW_HEIGHT_PX * dpi / 72.0 * 0.85))
        ax_heat.set_yticks(region_tick_indices)
        ax_heat.set_yticklabels(region_labels, fontsize=ytick_fontsize)
        ax_heat.set_ylabel(item_label)

        sig_y, sig_x = np.where(snapshot.significant_mask)
        if sig_x.size:
            ax_heat.scatter(
                sig_x,
                sig_y,
                s=16,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.6,
                label="significant bin",
            )
            ax_heat.legend(loc="upper right", fontsize=7)
        metric_label = "t-value" if heatmap_metric == "t_value" else "mean difference"
        ax_heat.set_title(f"{metric_label.capitalize()} heatmap with significant-bin markers")

    if ax_table is not None:
        ax_table.axis("off")
        if has_multi_bin:
            col_labels = [
                item_label,
                f"{cond_a}_mean",
                f"{cond_b}_mean",
                "delta_mean",
                "p_min_corr",
                "p_min_raw",
                "t@p_min",
                "sem_diff@pmin",
                "ci95_diff@pmin",
                f"sig_bins/{snapshot.effective_n_bins}",
                "sig_any",
            ]
            cell_text = [
                [
                    row.label,
                    _format_float(row.mean_a_global),
                    _format_float(row.mean_b_global),
                    _format_float(row.delta_mean_global),
                    _format_float(row.p_min),
                    _format_float(row.p_raw_min),
                    _format_float(row.t_at_p_min),
                    _format_float(row.sem_diff_at_p_min),
                    f"[{_format_float(row.ci95_low_at_p_min)}, {_format_float(row.ci95_high_at_p_min)}]",
                    str(row.sig_bins),
                    "yes" if row.sig_any else "no",
                ]
                for row in metrics
            ]
        else:
            col_labels = [
                item_label,
                f"{cond_a}_mean",
                f"{cond_b}_mean",
                "delta_mean",
                "p_corr",
                "p_raw",
                "t",
                "sem_diff",
                "ci95_diff",
                "significant",
            ]
            cell_text = [
                [
                    row.label,
                    _format_float(row.mean_a_global),
                    _format_float(row.mean_b_global),
                    _format_float(row.delta_mean_global),
                    _format_float(row.p_value),
                    _format_float(row.p_raw),
                    _format_float(row.t_value),
                    _format_float(row.sem_diff),
                    f"[{_format_float(row.ci95_low)}, {_format_float(row.ci95_high)}]",
                    "yes" if row.significant else "no",
                ]
                for row in metrics
            ]

        if not cell_text:
            cell_text = [["n/a"] + [""] * (len(col_labels) - 1)]

        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="lower center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table_font_size = 7 if len(metrics) <= 16 else (6 if len(metrics) <= 30 else 5)
        table.set_fontsize(table_font_size)
        table.scale(1.0, 1.0)

    if snapshot.binning_mode == "window_ms":
        binning_line = (
            f"binning: window_ms={snapshot.window_ms:.3g}, "
            f"effective_n_bins={snapshot.effective_n_bins}"
        )
    elif snapshot.binning_mode == "n_bins":
        binning_line = (
            f"binning: n_bins={snapshot.n_bins}, "
            f"effective_n_bins={snapshot.effective_n_bins}"
        )
    else:
        binning_line = f"binning: none, effective_n_bins={snapshot.effective_n_bins}"

    params_lines = [
        f"file: {snapshot.stats_path.name}",
        f"stats_valid: {snapshot.stats_valid}",
        f"conditions: {cond_a}={n_a}, {cond_b}={n_b}",
        f"correction: {snapshot.correction_method}",
        f"alpha: {snapshot.significance_alpha:.4g}",
        "uncertainty: sem_diff + ci95 stored in HDF5",
        binning_line,
        f"analysis_level: {snapshot.analysis_level}",
        f"atlas_name: {snapshot.atlas_name or 'n/a'}",
    ]
    if snapshot.pipeline_name:
        params_lines.append(f"pipeline: {snapshot.pipeline_name}")
    if snapshot.primary_condition_metric:
        params_lines.append(f"source_metric: {snapshot.primary_condition_metric}")
    if snapshot.roi_mode:
        params_lines.append(f"roi_mode: {snapshot.roi_mode}")
    if (
        snapshot.roi_channel_counts.size == len(snapshot.channels)
        and snapshot.roi_subject_counts.size == len(snapshot.channels)
        and len(snapshot.channels) > 0
    ):
        params_lines.append(
            "roi_counts: "
            f"channels[min={int(np.min(snapshot.roi_channel_counts))}, "
            f"max={int(np.max(snapshot.roi_channel_counts))}], "
            f"subjects[min={int(np.min(snapshot.roi_subject_counts))}, "
            f"max={int(np.max(snapshot.roi_subject_counts))}]"
        )
    params_lines.extend(_region_channel_panel_lines(snapshot))

    right_margin = 0.65 if show_params_panel else 0.97
    suptitle_x = 0.38 if show_params_panel else 0.5
    fig.suptitle(
        f"Trial-Stats Summary | {snapshot.stats_path.stem}",
        fontsize=12,
        x=suptitle_x,
    )
    fig.subplots_adjust(left=0.06, right=right_margin, top=0.92, bottom=0.07, hspace=0.6)
    if ax_table is not None:
        ax_table.text(
            0.5,
            0.94,
            f"Per-{item_label.lower()} statistics",
            transform=ax_table.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )
    if show_params_panel:
        fig.text(
            0.71,
            0.92,
            "\n".join(params_lines),
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            bbox={
                "boxstyle": "round",
                "facecolor": "#f7f7f7",
                "edgecolor": "#cfcfcf",
                "alpha": 0.95,
            },
        )

    out_path = _figure_output_path(snapshot.stats_path)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _inspect_file(
    stats_path: Path,
    top_k: int,
    save_channel_summary: bool,
    save_figure: bool,
    figure_dpi: int,
    show_heatmap: bool = True,
    show_bar_chart: bool = True,
    show_table: bool = True,
    show_params_panel: bool = True,
    heatmap_metric: str = "difference",
) -> None:
    snapshot = _load_snapshot(stats_path)
    trials_path = _trial_table_path(stats_path)
    trial_rows = _load_trial_rows(trials_path)
    exclusion_counts = _summarize_exclusions(trial_rows)

    n_channels = len(snapshot.channels)
    n_times = snapshot.p_values.shape[1] if snapshot.p_values.ndim == 2 else 0
    finite_p = int(np.isfinite(snapshot.p_values).sum())
    total_tests = int(snapshot.p_values.size)
    significant = int(snapshot.significant_mask.sum())

    ranking = _rank_channels(snapshot)
    top_ranking = ranking[:max(1, top_k)]

    out_channel_summary: Path | None = None
    if save_channel_summary and ranking:
        out_channel_summary = _write_channel_summary_tsv(stats_path, ranking)

    out_figure: Path | None = None
    if save_figure:
        out_figure = _write_summary_figure(
            snapshot,
            dpi=figure_dpi,
            show_heatmap=show_heatmap,
            show_bar_chart=show_bar_chart,
            show_table=show_table,
            show_params_panel=show_params_panel,
            heatmap_metric=heatmap_metric,
        )

    kept_fragment = ""
    if trial_rows:
        n_kept = sum(str(row.get("keep", "")).strip().lower() == "true" for row in trial_rows)
        kept_fragment = f", kept_trials={n_kept}/{len(trial_rows)}"

    axis_label = "regions" if _is_region_level(snapshot.analysis_level) else "channels"
    print(
        f"{stats_path.name}: "
        f"stats_valid={snapshot.stats_valid}, "
        f"{axis_label}={n_channels}, "
        f"bins={snapshot.effective_n_bins}, "
        f"finite_p={finite_p}/{total_tests}, "
        f"significant={significant}/{total_tests}"
        f"{kept_fragment}"
    )
    if top_ranking:
        best = top_ranking[0]
        item_label = "region" if _is_region_level(snapshot.analysis_level) else "channel"
        print(
            f"  best_{item_label}: {best['channel']} "
            f"(sig_fraction={float(best['sig_fraction']):.3f})"
        )
    if out_figure is not None:
        print(f"  figure: {out_figure}")
    else:
        print("  figure: disabled (--no-save-figure)")
    if out_channel_summary is not None:
        print(f"  channel_summary: {out_channel_summary}")
    if exclusion_counts:
        top_reason, top_count = next(iter(exclusion_counts.items()))
        print(f"  top_exclusion: {top_reason} ({top_count})")


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return sorted({path.resolve() for path in args.input})

    root = args.bids_root.resolve()
    return sorted({path.resolve() for path in root.glob(args.glob)})


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.figure_dpi < 1:
        raise ValueError("--figure-dpi must be >= 1")

    inputs = _resolve_inputs(args)
    if not inputs:
        print("No stats file found.")
        return

    print(f"Found {len(inputs)} trial-stats file(s).")
    for path in inputs:
        _inspect_file(
            stats_path=path,
            top_k=args.top_k,
            save_channel_summary=args.save_channel_summary,
            save_figure=args.save_figure,
            figure_dpi=args.figure_dpi,
            show_heatmap=SHOW_HEATMAP,
            show_bar_chart=SHOW_BAR_CHART,
            show_table=SHOW_TABLE,
            show_params_panel=SHOW_PARAMS_PANEL,
            heatmap_metric=HEATMAP_METRIC,
        )


if __name__ == "__main__":
    main()

