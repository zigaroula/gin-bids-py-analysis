"""
Export the group visualizer "Mean slope" plot to an image file.

Edit the parameters below, then run from the repository root:

    python scripts/export_trial_slope_group_mean.py

The plot is intentionally aligned with the "Group" -> "Mean slope" tab from
the interactive trial slope visualizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bidsforge.bids import BIDSDataset
from bidsforge.processing.trial_stats_group.regression.result_loader import (
    load_regression_group_result,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    EXPORT_CONDITION_A_COLOR as CONDITION_A_COLOR,
    EXPORT_CONDITION_B_COLOR as CONDITION_B_COLOR,
    EXPORT_FIGSIZE_INCHES as FIGSIZE_INCHES,
    EXPORT_OUTPUT_DIR as OUTPUT_DIR,
    EXPORT_OUTPUT_DPI as OUTPUT_DPI,
    EXPORT_OUTPUT_FILE as OUTPUT_FILE,
    EXPORT_OUTPUT_FORMAT as OUTPUT_FORMAT,
    EXPORT_ROI_NAME as ROI_NAME,
    EXPORT_SHOW_CONTRAST_SIGNIFICANCE_BAR as SHOW_CONTRAST_SIGNIFICANCE_BAR,
    EXPORT_SHOW_VS_ZERO_BOLD_SEGMENTS as SHOW_VS_ZERO_BOLD_SEGMENTS,
    EXPORT_TRANSPARENT as TRANSPARENT,
    EXPORT_X_LIMITS as X_LIMITS,
    EXPORT_Y_LIMITS as Y_LIMITS,
    GROUP_STATS_FILE,
    GROUP_STATS_FILTERS,
    print_recipe_summary,
)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def main() -> list[Path]:
    print_recipe_summary()
    group_stats_file = _resolve_group_stats_file()
    result = load_regression_group_result(group_stats_file)
    roi_names = _normalize_roi_names(ROI_NAME)
    print(f"Loaded:  {group_stats_file}")

    output_paths: list[Path] = []
    for requested_roi_name in roi_names:
        roi_idx = _find_roi_index(result.region_names, requested_roi_name)
        resolved_roi_name = str(result.region_names[roi_idx])
        output_path = _resolve_output_path(
            group_stats_file=group_stats_file,
            roi_name=resolved_roi_name,
            multiple_rois=len(roi_names) > 1,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = _draw_mean_slope_figure(result, roi_idx)
        fig.savefig(
            output_path,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            transparent=TRANSPARENT,
            format=OUTPUT_FORMAT,
        )
        plt.close(fig)

        print(f"ROI:     {resolved_roi_name}")
        print(f"Export:  {output_path}")
        output_paths.append(output_path)

    return output_paths


def _resolve_group_stats_file() -> Path:
    if GROUP_STATS_FILE is not None:
        path = Path(GROUP_STATS_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Group stats file not found: {path}")
        return path

    dataset = BIDSDataset(BIDS_ROOT)
    files = sorted(dataset.get_files(**GROUP_STATS_FILTERS), key=lambda file: str(file.path))
    if not files:
        raise FileNotFoundError(
            f"No group stats file found in {BIDS_ROOT} matching {GROUP_STATS_FILTERS}"
        )
    if len(files) > 1:
        print("Multiple group stats files matched; using the first one:")
        for file in files:
            print(f"  - {file.path}")
    return files[0].path


def _find_roi_index(region_names: list[str] | tuple[str, ...], roi_name: str) -> int:
    roi_name_cf = roi_name.casefold()
    for idx, name in enumerate(region_names):
        if str(name).casefold() == roi_name_cf:
            return idx
    available = ", ".join(str(name) for name in region_names)
    raise ValueError(f"ROI {roi_name!r} not found. Available ROIs: {available}")


def _normalize_roi_names(value: object) -> list[str]:
    if isinstance(value, str):
        roi_names = [value]
    else:
        roi_names = [str(item) for item in value]

    cleaned = [name.strip() for name in roi_names if name.strip()]
    if not cleaned:
        raise ValueError("EXPORT_ROI_NAME must contain at least one ROI name.")
    return cleaned


def _resolve_output_path(*, group_stats_file: Path, roi_name: str, multiple_rois: bool) -> Path:
    if OUTPUT_FILE is not None:
        path = Path(OUTPUT_FILE)
        suffix = path.suffix or f".{OUTPUT_FORMAT}"
        if multiple_rois:
            safe_roi = _safe_filename_part(roi_name)
            return path.with_name(f"{path.stem}_{safe_roi}{suffix}")
        if path.suffix:
            return path
        return path.with_suffix(suffix)

    desc = _extract_bids_desc(group_stats_file)
    safe_roi = _safe_filename_part(roi_name)
    safe_desc = _safe_filename_part(desc)
    filename = f"group_mean_slope_{safe_roi}_{safe_desc}.{OUTPUT_FORMAT}"
    return OUTPUT_DIR / filename


def _draw_mean_slope_figure(result: object, roi_idx: int) -> plt.Figure:
    time_axis = np.asarray(result.time_axis_s, dtype=np.float64)
    roi_label = str(result.region_names[roi_idx])
    condition_a_label, condition_b_label = _condition_labels(result)
    metric_label = _regression_metric_label(result)
    metric_estimates = _regression_metric_estimates(result)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES, tight_layout=True)

    has_slope_mean = (
        np.asarray(metric_estimates.condition_a.mean).size > 0
        and np.asarray(metric_estimates.condition_b.mean).size > 0
    )
    if not has_slope_mean:
        ax.text(
            0.5,
            0.5,
            f"No {metric_label} means available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
        )
    else:
        slope_mean_a = np.asarray(metric_estimates.condition_a.mean[roi_idx], dtype=np.float64)
        slope_sem_a = np.asarray(metric_estimates.condition_a.sem[roi_idx], dtype=np.float64)
        slope_mean_b = np.asarray(metric_estimates.condition_b.mean[roi_idx], dtype=np.float64)
        slope_sem_b = np.asarray(metric_estimates.condition_b.sem[roi_idx], dtype=np.float64)

        ax.plot(time_axis, slope_mean_a, color=CONDITION_A_COLOR, label=condition_a_label)
        ax.fill_between(
            time_axis,
            slope_mean_a - slope_sem_a,
            slope_mean_a + slope_sem_a,
            alpha=0.25,
            color=CONDITION_A_COLOR,
        )
        ax.plot(time_axis, slope_mean_b, color=CONDITION_B_COLOR, label=condition_b_label)
        ax.fill_between(
            time_axis,
            slope_mean_b - slope_sem_b,
            slope_mean_b + slope_sem_b,
            alpha=0.25,
            color=CONDITION_B_COLOR,
        )

        if SHOW_VS_ZERO_BOLD_SEGMENTS:
            sig_a_vz = _roi_bool_mask(
                result.regression_stats.vs_zero.condition_a.significant_mask,
                roi_idx,
                len(time_axis),
            )
            sig_b_vz = _roi_bool_mask(
                result.regression_stats.vs_zero.condition_b.significant_mask,
                roi_idx,
                len(time_axis),
            )
            if sig_a_vz.any():
                ax.plot(
                    time_axis,
                    np.ma.array(slope_mean_a, mask=~sig_a_vz),
                    color=CONDITION_A_COLOR,
                    linewidth=4.5,
                )
            if sig_b_vz.any():
                ax.plot(
                    time_axis,
                    np.ma.array(slope_mean_b, mask=~sig_b_vz),
                    color=CONDITION_B_COLOR,
                    linewidth=4.5,
                )

        all_slopes = np.concatenate([slope_mean_a, slope_mean_b])
        if np.nanmin(all_slopes) < 0 < np.nanmax(all_slopes):
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"mean {metric_label}")
    ax.set_title(f"{_title_base(result, roi_idx)} - {metric_label} mean", fontsize=9)

    if SHOW_CONTRAST_SIGNIFICANCE_BAR:
        sig_slope = _roi_bool_mask(
            result.regression_stats.contrast.significant_mask,
            roi_idx,
            len(time_axis),
        )
        if sig_slope.any():
            ax.fill_between(
                time_axis,
                0.005,
                0.025,
                where=sig_slope,
                alpha=0.75,
                color="red",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )

    _safe_legend(ax)
    if X_LIMITS is not None:
        ax.set_xlim(*X_LIMITS)
    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)
    else:
        _set_symmetric_ylim(ax)
    return fig


def _roi_bool_mask(mask_array: np.ndarray, roi_idx: int, n_times: int) -> np.ndarray:
    arr = np.asarray(mask_array)
    if arr.size == 0:
        return np.zeros(n_times, dtype=bool)
    return np.asarray(arr[roi_idx], dtype=bool).reshape(n_times)


def _title_base(result: object, roi_idx: int) -> str:
    n_channels = _safe_count(result.roi_channel_counts, roi_idx)
    n_subjects = _safe_count(result.roi_subject_counts, roi_idx)
    return f"{result.region_names[roi_idx]}  -  {n_channels} channel(s) / {n_subjects} subject(s)"


def _safe_count(values: np.ndarray, idx: int) -> str:
    arr = np.asarray(values)
    if arr.size <= idx:
        return "?"
    return str(int(arr[idx]))


def _condition_labels(result: object) -> tuple[str, str]:
    labels = tuple(getattr(result, "condition_labels", ("condition_a", "condition_b")))
    condition_a = str(labels[0]) if len(labels) >= 1 else "condition_a"
    condition_b = str(labels[1]) if len(labels) >= 2 else "condition_b"
    return condition_a, condition_b


def _regression_metric_label(result: object) -> str:
    metric = str(getattr(result, "primary_regression_metric", "slope")).strip().lower()
    if metric == "r_value":
        return "r"
    return "slope"


def _regression_metric_estimates(result: object) -> object:
    metric = str(getattr(result, "primary_regression_metric", "slope")).strip().lower()
    return getattr(result, "r_value") if metric == "r_value" else getattr(result, "slope")


def _set_symmetric_ylim(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    bound = max(abs(ymin), abs(ymax))
    if bound > 0:
        ax.set_ylim(-bound, bound)


def _safe_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize="small", loc="upper right")


def _extract_bids_desc(path: Path) -> str:
    for part in path.stem.split("_"):
        if part.startswith("desc-"):
            return part.removeprefix("desc-")
    return "stats"


def _safe_filename_part(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in text)
    return safe.strip("_") or "value"


if __name__ == "__main__":
    main()

