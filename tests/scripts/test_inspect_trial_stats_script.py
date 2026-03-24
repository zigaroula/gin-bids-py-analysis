from __future__ import annotations

import subprocess
import shutil
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_trial_stats.py"


def _trial_table_path(stats_path: Path) -> Path:
    tsv_path = stats_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


def _write_trial_table(stats_path: Path) -> None:
    trial_path = _trial_table_path(stats_path)
    trial_path.write_text(
        "\n".join(
            [
                (
                    "source_file\tanchor_event_index\tanchor_event_code\tanchor_onset_s\t"
                    "trial_id\tresolved_label\tkeep\texclusion_reason"
                ),
                (
                    f"{stats_path}\t0\t10\t1.0\ttrial-000\taccepted\ttrue\t"
                ),
                (
                    f"{stats_path}\t1\t10\t2.0\ttrial-001\trejected\tfalse\tunsupported_label"
                ),
            ]
        ),
        encoding="utf-8",
    )


def _make_case_dir(case_name: str) -> Path:
    root = REPO_ROOT / "tests" / "_script_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{case_name}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _write_stats_h5(
    path: Path,
    *,
    n_regions: int,
    n_bins: int,
    include_optional: bool = True,
    include_significant_mask: bool = True,
    include_trial_labels: bool = True,
    include_modern_binning_meta: bool = True,
) -> None:
    time_s = np.linspace(-0.2, 0.5, num=n_bins, dtype=np.float64)
    t_values = np.linspace(1.0, 2.5, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    p_values = np.linspace(0.01, 0.20, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    significant_mask = p_values < 0.05

    accepted = np.linspace(10.0, 15.0, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    rejected = accepted - 1.5
    difference = accepted - rejected

    with h5py.File(path, "w") as fh:
        stats_grp = fh.create_group("stats")
        stats_grp.create_dataset("t_values", data=t_values)
        stats_grp.create_dataset("p_values", data=p_values)
        if include_optional:
            stats_grp.create_dataset("p_values_uncorrected", data=np.minimum(1.0, p_values * 1.1))
        if include_significant_mask:
            stats_grp.create_dataset("significant_mask", data=significant_mask)

        means_grp = fh.create_group("means")
        means_grp.create_dataset("accepted", data=accepted)
        means_grp.create_dataset("rejected", data=rejected)
        means_grp.create_dataset("difference", data=difference)

        uncertainty_grp = fh.create_group("uncertainty")
        accepted_sem = np.full_like(accepted, 0.25, dtype=np.float64)
        rejected_sem = np.full_like(rejected, 0.20, dtype=np.float64)
        difference_sem = np.sqrt((accepted_sem ** 2) + (rejected_sem ** 2))
        uncertainty_grp.create_dataset("accepted_sem", data=accepted_sem)
        uncertainty_grp.create_dataset("rejected_sem", data=rejected_sem)
        uncertainty_grp.create_dataset("difference_sem", data=difference_sem)
        uncertainty_grp.create_dataset("difference_ci95_low", data=difference - (1.96 * difference_sem))
        uncertainty_grp.create_dataset("difference_ci95_high", data=difference + (1.96 * difference_sem))

        axes_grp = fh.create_group("axes")
        axes_grp.create_dataset(
            "region",
            data=np.array([f"ROI_{idx:02d}" for idx in range(n_regions)], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        axes_grp.create_dataset("time_s", data=time_s)

        meta_grp = fh.create_group("meta")
        meta_grp.create_dataset("analysis_level", data="roi", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("trial_counts", data=np.array([12, 11], dtype=np.int64))
        if include_trial_labels:
            meta_grp.create_dataset(
                "trial_count_labels",
                data=np.array(["accepted", "rejected"], dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        meta_grp.create_dataset("p_value_correction_method", data="fdr_bh", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("significance_alpha", data=0.05)
        meta_grp.create_dataset("atlas_name", data="MarsAtlas", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset(
            "atlas_regions",
            data=np.array([f"ROI_{idx:02d}" for idx in range(n_regions)], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        atlas_map_grp = meta_grp.create_group("atlas_region_channel_map")
        region_order = [f"ROI_{idx:02d}" for idx in range(n_regions)]
        atlas_map_grp.create_dataset(
            "region_order",
            data=np.array(region_order, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        map_regions = []
        map_channels = []
        for idx, region in enumerate(region_order):
            map_regions.extend([region, region])
            map_channels.extend([f"CH{idx + 1:02d}a", f"CH{idx + 1:02d}b"])
        atlas_map_grp.create_dataset(
            "region",
            data=np.array(map_regions, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        atlas_map_grp.create_dataset(
            "channel",
            data=np.array(map_channels, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        meta_grp.create_dataset("stats_valid", data=True)

        if include_modern_binning_meta:
            meta_grp.create_dataset("window_ms", data=0.0)
            meta_grp.create_dataset("n_bins", data=n_bins)
            meta_grp.create_dataset("effective_n_bins", data=n_bins)
            meta_grp.create_dataset(
                "binning_mode",
                data="n_bins" if n_bins > 1 else "none",
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        else:
            meta_grp.create_dataset("temporal_window_ms", data=0.0)


def _write_group_stats_h5(path: Path, *, n_regions: int, n_bins: int) -> None:
    time_s = np.linspace(-0.2, 0.5, num=n_bins, dtype=np.float64)
    t_values = np.linspace(1.0, 2.0, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    p_values = np.linspace(0.01, 0.20, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    significant_mask = p_values < 0.05
    metric_mean = np.linspace(0.2, 1.0, num=n_regions * n_bins, dtype=np.float64).reshape(
        n_regions,
        n_bins,
    )
    metric_sem = np.full_like(metric_mean, 0.15, dtype=np.float64)

    with h5py.File(path, "w") as fh:
        stats_grp = fh.create_group("stats")
        stats_grp.create_dataset("t_values", data=t_values)
        stats_grp.create_dataset("p_values", data=p_values)
        stats_grp.create_dataset("p_values_uncorrected", data=p_values)
        stats_grp.create_dataset("significant_mask", data=significant_mask)

        means_grp = fh.create_group("means")
        means_grp.create_dataset("metric_mean", data=metric_mean)

        uncertainty_grp = fh.create_group("uncertainty")
        uncertainty_grp.create_dataset("metric_sem", data=metric_sem)

        summary_grp = fh.create_group("summary_epoch")
        summary_grp.create_dataset("t_values", data=np.linspace(2.0, 3.0, num=n_regions))
        summary_grp.create_dataset("p_values", data=np.linspace(0.01, 0.04, num=n_regions))
        summary_grp.create_dataset("df", data=np.full((n_regions,), 12.0, dtype=np.float64))
        summary_grp.create_dataset("metric_mean", data=np.linspace(0.4, 0.9, num=n_regions))
        summary_grp.create_dataset("metric_sem", data=np.full((n_regions,), 0.1, dtype=np.float64))
        summary_grp.create_dataset("roi_channel_counts", data=np.array([4] * n_regions, dtype=np.int64))
        summary_grp.create_dataset("roi_subject_counts", data=np.array([3] * n_regions, dtype=np.int64))

        axes_grp = fh.create_group("axes")
        region_names = [f"ROI_{idx:02d}" for idx in range(n_regions)]
        axes_grp.create_dataset(
            "region",
            data=np.array(region_names, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        axes_grp.create_dataset("time_s", data=time_s)

        meta_grp = fh.create_group("meta")
        meta_grp.create_dataset("analysis_level", data="roi_group", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("source_metric", data="mean_difference", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("condition_labels", data=np.array(["accepted", "rejected"], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("p_value_correction_method", data="none", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("significance_alpha", data=0.05)
        meta_grp.create_dataset("roi_mode", data="manual", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("atlas_name", data="", dtype=h5py.string_dtype(encoding="utf-8"))
        meta_grp.create_dataset("window_ms", data=0.0)
        meta_grp.create_dataset("n_bins", data=n_bins)
        meta_grp.create_dataset("effective_n_bins", data=n_bins)
        meta_grp.create_dataset("binning_mode", data="n_bins", dtype=h5py.string_dtype(encoding="utf-8"))

        excluded_grp = meta_grp.create_group("excluded_rois")
        excluded_grp.create_dataset("region", data=np.array([], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        excluded_grp.create_dataset("reason", data=np.array([], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))

        contrib_grp = fh.create_group("contributions")
        contrib_regions = []
        contrib_subjects = []
        contrib_channels = []
        for idx, region in enumerate(region_names):
            contrib_regions.extend([region, region])
            contrib_subjects.extend(["01", "02"])
            contrib_channels.extend([f"A{idx+1}", f"B{idx+1}"])
        contrib_grp.create_dataset("region", data=np.array(contrib_regions, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        contrib_grp.create_dataset("subject", data=np.array(contrib_subjects, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        contrib_grp.create_dataset("channel", data=np.array(contrib_channels, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        contrib_grp.create_dataset("source_stats_file", data=np.array([str(path)] * len(contrib_regions), dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))

        prov_grp = fh.create_group("provenance")
        prov_grp.create_dataset("source_trial_stats_files", data=np.array([str(path)], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        prov_grp.create_dataset("source_electrodes_files", data=np.array([], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        prov_grp.create_dataset("pipeline_name", data="trial_stats_group", dtype=h5py.string_dtype(encoding="utf-8"))
        prov_grp.create_dataset("pipeline_version", data="test", dtype=h5py.string_dtype(encoding="utf-8"))


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT_PATH), *args]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_inspect_trial_stats_generates_png_for_single_bin() -> None:
    case_dir = _make_case_dir("single_bin")
    try:
        stats_path = case_dir / "sub-01_desc-trialstats_stats.h5"
        _write_stats_h5(stats_path, n_regions=3, n_bins=1)
        _write_trial_table(stats_path)

        result = _run_script("--input", str(stats_path), "--figure-dpi", "90")
        figure_path = stats_path.with_name(f"{stats_path.stem}_summary.png")

        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
        assert "stats_valid=True" in result.stdout
        assert f"figure: {figure_path}" in result.stdout
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_inspect_trial_stats_generates_png_for_multibin_case() -> None:
    case_dir = _make_case_dir("multi_bin")
    try:
        stats_path = case_dir / "sub-02_desc-trialstats_stats.h5"
        _write_stats_h5(stats_path, n_regions=4, n_bins=5)
        _write_trial_table(stats_path)

        result = _run_script("--input", str(stats_path), "--figure-dpi", "90")
        figure_path = stats_path.with_name(f"{stats_path.stem}_summary.png")

        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
        assert "bins=5" in result.stdout
        assert f"figure: {figure_path}" in result.stdout
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_inspect_trial_stats_handles_missing_optional_fields_and_no_save_figure() -> None:
    case_dir = _make_case_dir("legacy_fields")
    try:
        stats_path = case_dir / "sub-03_desc-trialstats_stats.h5"
        _write_stats_h5(
            stats_path,
            n_regions=2,
            n_bins=3,
            include_optional=False,
            include_significant_mask=False,
            include_trial_labels=False,
            include_modern_binning_meta=False,
        )
        _write_trial_table(stats_path)

        result = _run_script("--input", str(stats_path), "--no-save-figure")
        figure_path = stats_path.with_name(f"{stats_path.stem}_summary.png")

        assert not figure_path.exists()
        assert "figure: disabled (--no-save-figure)" in result.stdout
        assert "Found 1 trial-stats file(s)." in result.stdout
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_inspect_trial_stats_supports_group_level_output() -> None:
    case_dir = _make_case_dir("group_level")
    try:
        stats_path = case_dir / "sub-group_task-decid_desc-trialstatsgroup_stats.h5"
        _write_group_stats_h5(stats_path, n_regions=2, n_bins=4)

        result = _run_script("--input", str(stats_path), "--figure-dpi", "90")
        figure_path = stats_path.with_name(f"{stats_path.stem}_summary.png")

        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
        assert "regions=2" in result.stdout
        assert "bins=4" in result.stdout
        assert f"figure: {figure_path}" in result.stdout
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


