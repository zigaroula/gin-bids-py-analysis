"""
Inspect trial-stats outputs (.h5 + companion _trials.tsv).

Examples:
  python scripts/inspect_trial_stats.py --input E:\\CBT\\bids\\derivatives\\trial_stats\\sub-01\\ieeg\\sub-01_desc-trialstats_stats.h5
  python scripts/inspect_trial_stats.py --bids-root E:\\CBT\\bids
  python scripts/inspect_trial_stats.py --bids-root E:\\CBT\\bids --save-channel-summary
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_BIDS_ROOT = Path(r"D:\CBT\bids")
DEFAULT_STATS_GLOB = "derivatives/trial_stats/**/*_stats.h5"


@dataclass
class TrialStatsSnapshot:
    stats_path: Path
    channels: list[str]
    time_s: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    p_values_uncorrected: np.ndarray
    significant_mask: np.ndarray
    mean_difference: np.ndarray
    condition_labels: tuple[str, str]
    trial_counts: tuple[int, int]
    stats_valid: bool
    correction_method: str
    significance_alpha: float
    analysis_level: str
    atlas_name: str
    atlas_regions: list[str]
    window_ms: float
    n_bins: int
    effective_n_bins: int
    binning_mode: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect trial-stats outputs and print a QC summary.",
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
        help="Number of top channels to display (default: %(default)s)",
    )
    parser.add_argument(
        "--save-channel-summary",
        action="store_true",
        help="Write a <stem>_channel_summary.tsv next to each stats file.",
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


def _bool_scalar(dataset: h5py.Dataset, default: bool = False) -> bool:
    try:
        return bool(dataset[()])
    except Exception:
        return default


def _float_scalar(dataset: h5py.Dataset, default: float = np.nan) -> float:
    try:
        return float(dataset[()])
    except Exception:
        return default


def _str_scalar(dataset: h5py.Dataset, default: str = "") -> str:
    try:
        return str(dataset.asstr()[()])
    except Exception:
        return default


def _load_snapshot(stats_path: Path) -> TrialStatsSnapshot:
    with h5py.File(stats_path, "r") as fh:
        stats_grp = fh["stats"]
        axes_grp = fh["axes"]
        meta_grp = fh["meta"]
        means_grp = fh["means"]

        channels = _decode_str_array(np.asarray(axes_grp["channel"][:]))
        time_s = np.asarray(axes_grp["time_s"][:], dtype=np.float64)

        t_values = np.asarray(stats_grp["t_values"][:], dtype=np.float64)
        p_values = np.asarray(stats_grp["p_values"][:], dtype=np.float64)
        p_values_uncorrected = np.asarray(
            stats_grp["p_values_uncorrected"][:],
            dtype=np.float64,
        ) if "p_values_uncorrected" in stats_grp else p_values.copy()

        significance_alpha = _float_scalar(meta_grp.get("significance_alpha"), default=0.05)
        correction_method = _str_scalar(
            meta_grp.get("p_value_correction_method"),
            default="unknown",
        )
        significant_mask = np.asarray(
            stats_grp["significant_mask"][:],
            dtype=bool,
        ) if "significant_mask" in stats_grp else (
            np.isfinite(p_values) & (p_values < significance_alpha)
        )

        mean_difference = np.asarray(means_grp["difference"][:], dtype=np.float64)

        if "trial_count_labels" in meta_grp:
            raw_labels = _decode_str_array(np.asarray(meta_grp["trial_count_labels"][:]))
            if len(raw_labels) >= 2:
                condition_labels = (raw_labels[0], raw_labels[1])
            else:
                condition_labels = ("condition_a", "condition_b")
        else:
            condition_labels = ("condition_a", "condition_b")

        if "trial_counts" in meta_grp:
            raw_counts = np.asarray(meta_grp["trial_counts"][:], dtype=np.int64)
            if raw_counts.size >= 2:
                trial_counts = (int(raw_counts[0]), int(raw_counts[1]))
            else:
                trial_counts = (0, 0)
        else:
            trial_counts = (0, 0)

        stats_valid = _bool_scalar(meta_grp.get("stats_valid"), default=False)
        analysis_level = _str_scalar(meta_grp.get("analysis_level"), default="channel")
        atlas_name = _str_scalar(meta_grp.get("atlas_name"), default="")
        atlas_regions = (
            _decode_str_array(np.asarray(meta_grp["atlas_regions"][:]))
            if "atlas_regions" in meta_grp
            else []
        )
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
        mean_difference=mean_difference,
        condition_labels=condition_labels,
        trial_counts=trial_counts,
        stats_valid=stats_valid,
        correction_method=correction_method,
        significance_alpha=significance_alpha,
        analysis_level=analysis_level,
        atlas_name=atlas_name,
        atlas_regions=atlas_regions,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        binning_mode=binning_mode,
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


def _inspect_file(stats_path: Path, top_k: int, save_channel_summary: bool) -> None:
    snapshot = _load_snapshot(stats_path)
    trials_path = _trial_table_path(stats_path)
    trial_rows = _load_trial_rows(trials_path)
    exclusion_counts = _summarize_exclusions(trial_rows)

    n_channels, n_times = snapshot.p_values.shape if snapshot.p_values.ndim == 2 else (0, 0)
    finite_p = int(np.isfinite(snapshot.p_values).sum())
    total_tests = int(snapshot.p_values.size)
    significant = int(snapshot.significant_mask.sum())

    cond_a, cond_b = snapshot.condition_labels
    n_a, n_b = snapshot.trial_counts

    print("=" * 80)
    print(stats_path)
    print(f"stats_valid={snapshot.stats_valid}")
    print(
        f"conditions: {cond_a}={n_a} trial(s), {cond_b}={n_b} trial(s)"
    )
    print(
        "p-values: method="
        f"{snapshot.correction_method}, alpha={snapshot.significance_alpha:.4g}, "
        f"finite={finite_p}/{total_tests}, significant={significant}/{total_tests}"
    )
    if snapshot.binning_mode == "window_ms":
        binning_info = (
            f"mode=window_ms, window_ms={snapshot.window_ms:.3f}, "
            f"effective_n_bins={snapshot.effective_n_bins}"
        )
    elif snapshot.binning_mode == "n_bins":
        binning_info = (
            f"mode=n_bins, n_bins={snapshot.n_bins}, "
            f"effective_n_bins={snapshot.effective_n_bins}"
        )
    else:
        binning_info = f"mode=none, effective_n_bins={snapshot.effective_n_bins}"
    print("analysis: " f"level={snapshot.analysis_level}, {binning_info}")
    if snapshot.analysis_level == "roi":
        print(
            "atlas: "
            f"name={snapshot.atlas_name or 'n/a'}, regions={len(snapshot.atlas_regions)}"
        )
    print(f"shape: channels={n_channels}, times={n_times}")

    if trial_rows:
        n_kept = sum(str(row.get("keep", "")).strip().lower() == "true" for row in trial_rows)
        print(f"trials table: {len(trial_rows)} row(s), kept={n_kept}, excluded={len(trial_rows) - n_kept}")
        if exclusion_counts:
            print("top exclusion reasons:")
            for reason, count in list(exclusion_counts.items())[:5]:
                print(f"  - {reason}: {count}")
    else:
        print(f"trials table: not found ({trials_path.name})")

    ranking = _rank_channels(snapshot)
    if ranking:
        item_label = "region" if snapshot.analysis_level == "roi" else "channel"
        print(f"top {min(top_k, len(ranking))} {item_label}(s):")
        for idx, row in enumerate(ranking[:top_k], start=1):
            print(
                f"  {idx:02d}. {row['channel']}: "
                f"sig_fraction={float(row['sig_fraction']):.3f}, "
                f"min_p={float(row['min_p']):.3g}, "
                f"min_p_raw={float(row['min_p_uncorrected']):.3g}, "
                f"max|t|={float(row['max_abs_t']):.3g}, "
                f"max|diff|={float(row['max_abs_diff']):.3g}"
            )

        if save_channel_summary:
            out_path = _write_channel_summary_tsv(stats_path, ranking)
            print(f"channel summary written: {out_path}")


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


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return sorted({path.resolve() for path in args.input})

    root = args.bids_root.resolve()
    return sorted({path.resolve() for path in root.glob(args.glob)})


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

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
        )


if __name__ == "__main__":
    main()
