"""
Run two accepted-vs-rejected condition tests and export channel/group figures.

Run from the repository root:

    python scripts/debug/run_onset_description_condition_tests.py

Outputs:
- subject-level condition-test files under derivatives/condition_test
- one curve per channel of interest under derivatives/condition_test/figures
- group-level condition-test files under derivatives/condition_test_group
- one group curve per ROI of interest under derivatives/condition_test_group/figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingResult,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
)
from bidsforge.processing.trial_stats_group import (
    ConditionTestGroupParams,
    ConditionTestGroupProcessing,
    ConditionTestGroupProcessingResult,
    ConditionTestGroupProcessingWriter,
    ConditionTestGroupWriterParams,
)
from bidsforge.visualization.trial_stats._bridge import group_file_group_context

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "accepted",
            "when": {"column": "choice", "op": "==", "value": "1"},
        },
        {
            "label": "rejected",
            "when": {"column": "choice", "op": "==", "value": "0"},
        },
    ],
)

MANUAL_REGION_CHANNELS = {
    "daINS": {
        "epi01": ["Y02", "Y06"],
        "epi03": ["IAD2"],
        "epi05": ["X04", "X07", "X03", "X06"],
        "epi07": ["X08", "T03"],
        "epi11": ["X05", "X07", "X06"],
        "epi12": ["Ap02"],
        "epi14": ["Xp04"],
        "epi17": ["II8"],
        "epi18": ["X06"],
        "epi19": ["IMD2"],
        "epi21": ["XS7", "XD7", "XD4", "XD8"],
        "epi22": ["XS8"],
        "epi23": ["EL2"],
    },
    "vaINS": {
        "epi04": ["Bp02"],
        "epi08": ["IA2", "IA4"],
        "epi11": ["X04"],
        "epi14": ["Xp02"],
        "epi17": ["II2"],
        "epi18": ["Y02", "X02"],
        "epi21": ["XS4"],
        "epi22": ["XS2", "YS2"],
    },
}

N_JOBS = 1
OUTPUT_FORMAT = "hdf5"
PLOT_OUTPUT_DIR = BIDS_ROOT / "derivatives" / "condition_test" / "figures"
GROUP_PLOT_OUTPUT_DIR = BIDS_ROOT / "derivatives" / "condition_test_group" / "figures"
PLOT_DPI = 160
CONDITION_A_COLOR = "steelblue"
CONDITION_B_COLOR = "tomato"
GROUP_ROI_NAMES = ("daINS", "vaINS")


@dataclass(frozen=True)
class AnalysisRecipe:
    label: str
    output_description: str
    anchor_event_code: str
    tmin_s: float
    tmax_s: float


RECIPES = [
    AnalysisRecipe(
        label="event10_onset",
        output_description="onset",
        anchor_event_code="10",
        tmin_s=-1.0,
        tmax_s=6.0,
    ),
    AnalysisRecipe(
        label="event15_response",
        output_description="response",
        anchor_event_code="15",
        tmin_s=-6.0,
        tmax_s=1.0,
    ),
]


def main() -> None:
    dataset = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    all_written_stats: list[Path] = []
    all_written_exports: list[Path] = []
    all_written_group_stats: list[Path] = []

    for recipe in RECIPES:
        print("")
        print(
            f"Running {recipe.label}: event {recipe.anchor_event_code}, "
            f"{recipe.tmin_s:g} to {recipe.tmax_s:g}s, desc-{recipe.output_description}"
        )
        params = _build_params(recipe)
        processor = ConditionTestProcessing(params, resolver=RESOLVER)
        writer = ConditionTestProcessingWriter(
            ConditionTestWriterParams(
                bids_root=BIDS_ROOT,
                output_format=OUTPUT_FORMAT,
                output_description=recipe.output_description,
            )
        )
        writer.write_dataset_description(params)

        results = processor.execute(groups, n_jobs=N_JOBS)
        subject_results: dict[str, ConditionTestProcessingResult] = {}
        for result in results:
            stats_path = writer.write(result)
            all_written_stats.append(stats_path)
            print(f"Wrote {stats_path}")

            subject = _subject_id(result)
            subject_results[subject] = result
            export_paths = _export_interest_curves(result, recipe, stats_path)
            all_written_exports.extend(export_paths)

        group_path, group_exports = _run_group_analysis(subject_results, recipe)
        if group_path is not None:
            all_written_group_stats.append(group_path)
        all_written_exports.extend(group_exports)

    print("")
    print(f"Wrote {len(all_written_stats)} stats file(s).")
    print(f"Wrote {len(all_written_group_stats)} group stats file(s).")
    print(f"Wrote {len(all_written_exports)} figure export file(s).")


def _build_params(recipe: AnalysisRecipe) -> ConditionTestParams:
    return ConditionTestParams(
        anchor_event_codes=[recipe.anchor_event_code],
        tmin_s=recipe.tmin_s,
        tmax_s=recipe.tmax_s,
        condition_a="accepted",
        condition_b="rejected",
        activity_zscore="baseline",
        activity_baseline_tmin_s=-0.25,
        activity_baseline_tmax_s=-0.05,
        activity_baseline_scope="global",
        activity_baseline_remove_outlier_trial_means=False,
        p_value_correction_method="none",
        n_permutations=0,
        significance_alpha=0.05,
    )


def _export_interest_curves(
    result: ConditionTestProcessingResult,
    recipe: AnalysisRecipe,
    stats_path: Path,
) -> list[Path]:
    subject = _subject_id(result)
    channel_to_idx = {name.casefold(): idx for idx, name in enumerate(result.channel_names)}
    output_paths: list[Path] = []

    for roi_name, subjects in MANUAL_REGION_CHANNELS.items():
        requested_channels = subjects.get(subject, [])

        for channel in requested_channels:
            idx = channel_to_idx.get(channel.casefold())
            if idx is None:
                print(f"Missing channel for {subject} {roi_name}: {channel}")
                continue

            channel_path = _plot_channel_curve(
                result=result,
                recipe=recipe,
                subject=subject,
                roi_name=roi_name,
                channel_name=result.channel_names[idx],
                channel_idx=idx,
            )
            output_paths.append(channel_path)

    if not output_paths:
        print(f"No interest-channel curves exported for {subject} ({stats_path.name}).")
    return output_paths


def _plot_channel_curve(
    *,
    result: ConditionTestProcessingResult,
    recipe: AnalysisRecipe,
    subject: str,
    roi_name: str,
    channel_name: str,
    channel_idx: int,
) -> Path:
    time_s = np.asarray(result.time_axis_s, dtype=np.float64)
    mean_a = np.asarray(result.signal_activity.condition_a.mean[channel_idx], dtype=np.float64)
    sem_a = np.asarray(result.signal_activity.condition_a.sem[channel_idx], dtype=np.float64)
    mean_b = np.asarray(result.signal_activity.condition_b.mean[channel_idx], dtype=np.float64)
    sem_b = np.asarray(result.signal_activity.condition_b.sem[channel_idx], dtype=np.float64)
    significant = np.asarray(result.contrast.significant_mask[channel_idx], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), tight_layout=True)
    _draw_condition_curves(ax, time_s, result, mean_a, sem_a, mean_b, sem_b)
    _shade_significance(ax, time_s, significant)
    ax.set_title(
        f"{subject} {roi_name} {channel_name} - desc-{recipe.output_description} "
        f"({result.condition_a_trial_count} x {result.condition_a} / "
        f"{result.condition_b_trial_count} x {result.condition_b})",
        fontsize=9,
    )
    _finish_axis(ax)

    output_path = (
        PLOT_OUTPUT_DIR
        / recipe.output_description
        / subject
        / f"{subject}_{recipe.output_description}_{roi_name}_{_safe_part(channel_name)}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_group_analysis(
    subject_results: dict[str, ConditionTestProcessingResult],
    recipe: AnalysisRecipe,
) -> tuple[Path | None, list[Path]]:
    if not subject_results:
        print(f"No subject results available for group analysis ({recipe.label}).")
        return None, []

    params = ConditionTestGroupParams(
        primary_condition_metric="mean_difference",
        p_value_correction_method="none",
        significance_alpha=0.05,
        roi_mode="manual",
        manual_region_channels=MANUAL_REGION_CHANNELS,
    )
    processor = ConditionTestGroupProcessing(params)
    writer = ConditionTestGroupProcessingWriter(
        ConditionTestGroupWriterParams(
            bids_root=BIDS_ROOT,
            output_format=OUTPUT_FORMAT,
            output_description=f"{recipe.output_description}group",
        )
    )
    writer.write_dataset_description(params)

    with group_file_group_context(
        subject_results,
        primary_condition_metric=params.primary_condition_metric,
    ) as group:
        result = processor.process_group(group)

    result.output_entities = _group_output_entities(subject_results)
    group_path = writer.write(result)
    print(f"Wrote group {group_path}")

    export_paths: list[Path] = []
    for roi_name in GROUP_ROI_NAMES:
        try:
            export_paths.append(_plot_group_roi_curve(result, recipe, roi_name))
        except ValueError as exc:
            print(f"Skipping group ROI export for {roi_name}: {exc}")
    return group_path, export_paths


def _plot_group_roi_curve(
    result: ConditionTestGroupProcessingResult,
    recipe: AnalysisRecipe,
    roi_name: str,
) -> Path:
    roi_idx = _find_roi_index(result.region_names, roi_name)
    time_s = np.asarray(result.time_axis_s, dtype=np.float64)
    mean_a = np.asarray(result.signal_activity.condition_a.mean[roi_idx], dtype=np.float64)
    sem_a = np.asarray(result.signal_activity.condition_a.sem[roi_idx], dtype=np.float64)
    mean_b = np.asarray(result.signal_activity.condition_b.mean[roi_idx], dtype=np.float64)
    sem_b = np.asarray(result.signal_activity.condition_b.sem[roi_idx], dtype=np.float64)
    significant = _roi_significance_mask(result, roi_idx, len(time_s))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), tight_layout=True)
    cond_a_label, cond_b_label = _condition_labels(result)
    _draw_labeled_condition_curves(
        ax,
        time_s,
        mean_a,
        sem_a,
        mean_b,
        sem_b,
        condition_a_label=cond_a_label,
        condition_b_label=cond_b_label,
    )
    _shade_significance(ax, time_s, significant)

    n_channels = _safe_count(result.roi_channel_counts, roi_idx)
    n_subjects = _safe_count(result.roi_subject_counts, roi_idx)
    ax.set_title(
        f"{roi_name} group - desc-{recipe.output_description} "
        f"({n_channels} channel(s) / {n_subjects} subject(s))",
        fontsize=9,
    )
    _finish_axis(ax)

    output_path = (
        GROUP_PLOT_OUTPUT_DIR
        / recipe.output_description
        / f"group_{recipe.output_description}_{_safe_part(roi_name)}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote group figure {output_path}")
    return output_path


def _draw_condition_curves(
    ax: plt.Axes,
    time_s: np.ndarray,
    result: ConditionTestProcessingResult,
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
) -> None:
    _draw_labeled_condition_curves(
        ax,
        time_s,
        mean_a,
        sem_a,
        mean_b,
        sem_b,
        condition_a_label=result.condition_a,
        condition_b_label=result.condition_b,
    )


def _draw_labeled_condition_curves(
    ax: plt.Axes,
    time_s: np.ndarray,
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
    *,
    condition_a_label: str,
    condition_b_label: str,
) -> None:
    ax.plot(time_s, mean_a, color=CONDITION_A_COLOR, label=condition_a_label)
    ax.fill_between(
        time_s,
        mean_a - sem_a,
        mean_a + sem_a,
        alpha=0.25,
        color=CONDITION_A_COLOR,
        linewidth=0,
    )
    ax.plot(time_s, mean_b, color=CONDITION_B_COLOR, label=condition_b_label)
    ax.fill_between(
        time_s,
        mean_b - sem_b,
        mean_b + sem_b,
        alpha=0.25,
        color=CONDITION_B_COLOR,
        linewidth=0,
    )


def _shade_significance(ax: plt.Axes, time_s: np.ndarray, significant: np.ndarray) -> None:
    if significant.size == time_s.size and significant.any():
        ax.fill_between(
            time_s,
            0.005,
            0.025,
            where=significant,
            alpha=0.75,
            color="red",
            transform=ax.get_xaxis_transform(),
            zorder=5,
        )


def _finish_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean activity")
    ax.legend(fontsize="small", loc="upper right")


def _subject_id(result: ConditionTestProcessingResult) -> str:
    subject = result.source_group.primary.get("subject")
    return str(subject) if subject else "unknown"


def _group_output_entities(
    subject_results: dict[str, ConditionTestProcessingResult],
) -> dict[str, str]:
    entities = {"subject": "group"}
    first_result = next(iter(subject_results.values()), None)
    if first_result is None:
        return entities

    task = first_result.source_group.primary.get("task")
    if task:
        entities["task"] = str(task)
    return entities


def _find_roi_index(region_names: list[str] | tuple[str, ...], roi_name: str) -> int:
    roi_name_cf = roi_name.casefold()
    for idx, name in enumerate(region_names):
        if str(name).casefold() == roi_name_cf:
            return idx
    available = ", ".join(str(name) for name in region_names)
    raise ValueError(f"ROI {roi_name!r} not found. Available ROIs: {available}")


def _roi_significance_mask(
    result: ConditionTestGroupProcessingResult,
    roi_idx: int,
    n_times: int,
) -> np.ndarray:
    mask = np.asarray(result.signal_activity_stats.significant_mask)
    if mask.size == 0 or mask.shape[0] <= roi_idx:
        return np.zeros(n_times, dtype=bool)
    return np.asarray(mask[roi_idx], dtype=bool).reshape(n_times)


def _condition_labels(result: ConditionTestGroupProcessingResult) -> tuple[str, str]:
    labels = tuple(result.condition_labels)
    condition_a = str(labels[0]) if len(labels) >= 1 else "condition_a"
    condition_b = str(labels[1]) if len(labels) >= 2 else "condition_b"
    return condition_a, condition_b


def _safe_count(values: np.ndarray, idx: int) -> str:
    arr = np.asarray(values)
    if arr.size <= idx:
        return "?"
    return str(int(arr[idx]))


def _safe_part(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in text)
    return safe.strip("_") or "value"


if __name__ == "__main__":
    main()
