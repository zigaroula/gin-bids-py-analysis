"""
Post-hoc correction for unpleasant trial-slope sign flips.

This script duplicates existing subject-level regression outputs for several
trial-slope descriptions, flips the condition-B/unpleasant slope sign back, then
recomputes group-level ROI results and exports mean-slope figures.

Run from the repository root:

    python scripts/correct_trial_slope_unpleasant_sign.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from gin_bids_py_analysis.bids import BIDSFile, BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupProcessing,
    RegressionGroupProcessingWriter,
    build_regression_compatible_groups,
)
from gin_bids_py_analysis.processing.trial_stats_group.regression.params import (
    RegressionGroupWriterParams,
)
from gin_bids_py_analysis.processing.trial_stats_group.regression.result_loader import (
    load_regression_group_result,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_trial_slope_group_mean import (  # noqa: E402
    _draw_mean_slope_figure,
    _find_roi_index,
    _safe_filename_part,
)
from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    EXPORT_OUTPUT_DIR,
    EXPORT_OUTPUT_DPI,
    EXPORT_OUTPUT_FORMAT,
    EXPORT_TRANSPARENT,
    TRIAL_SLOPE_GROUP_N_JOBS,
    build_group_params,
    print_roi_summary,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DESCRIPTION_MAP = {
    "onset": "onsetnoflip",
    "onsethfospikes": "onsethfospikesnoflip",
    "onset50hz": "onset50hznoflip",
}

EXPORT_ROIS = ["vmPFC", "aIns"]

SKIP_EXISTING = True
DRY_RUN = False
STOP_ON_ERROR = False

SUBJECT_REGRESSION_ROOT = BIDS_ROOT / "derivatives" / "regression"


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> list[Path]:
    params = build_group_params()
    print_roi_summary(params.manual_region_channels)

    written_paths: list[Path] = []
    for source_desc, target_desc in DESCRIPTION_MAP.items():
        print(f"\n=== {source_desc} -> {target_desc} ===")
        try:
            subject_outputs = correct_subject_files(
                source_desc=source_desc,
                target_desc=target_desc,
                search_root=SUBJECT_REGRESSION_ROOT,
                skip_existing=SKIP_EXISTING,
                dry_run=DRY_RUN,
            )
            written_paths.extend(subject_outputs)

            if DRY_RUN:
                continue

            group_outputs = recompute_group_results(
                subject_files=subject_outputs,
                target_desc=target_desc,
                group_params=params,
                skip_existing=SKIP_EXISTING,
            )
            written_paths.extend(group_outputs)

            for group_path in group_outputs:
                written_paths.extend(export_group_mean_slope_images(group_path, target_desc))
        except Exception as exc:
            message = f"Failed {source_desc} -> {target_desc}: {exc}"
            if STOP_ON_ERROR:
                raise RuntimeError(message) from exc
            print(message)

    return written_paths


def correct_subject_files(
    *,
    source_desc: str,
    target_desc: str,
    search_root: Path = SUBJECT_REGRESSION_ROOT,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> list[Path]:
    source_paths = find_subject_regression_files(search_root, source_desc)
    print(f"Found {len(source_paths)} subject file(s) for desc-{source_desc}.")

    out_paths: list[Path] = []
    for source_path in source_paths:
        target_path = path_with_replaced_desc(source_path, source_desc, target_desc)
        out_paths.append(target_path)

        if dry_run:
            print(f"[dry-run] {source_path} -> {target_path}")
            continue
        if skip_existing and target_path.exists():
            print(f"Skipping existing {target_path}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        correct_subject_hdf5(target_path, source_desc=source_desc, target_desc=target_desc)
        copy_and_correct_trial_tsv(source_path, target_path)
        print(f"Wrote {target_path}")

    return out_paths


def recompute_group_results(
    *,
    subject_files: Iterable[Path],
    target_desc: str,
    group_params: object,
    skip_existing: bool = True,
) -> list[Path]:
    files = [
        BIDSFile.from_path(path)
        for path in sorted(Path(path) for path in subject_files)
        if Path(path).exists()
    ]
    groups = build_regression_compatible_groups(files)
    print(
        f"Found {len(files)} corrected subject file(s) grouped into "
        f"{len(groups)} compatible group(s)."
    )
    if not groups:
        return []

    processor = RegressionGroupProcessing(group_params)
    writer = RegressionGroupProcessingWriter(
        RegressionGroupWriterParams(
            bids_root=BIDS_ROOT,
            output_format="hdf5",
            output_description=target_desc,
        )
    )
    out_paths = processor.run(
        groups,
        writer,
        n_jobs=TRIAL_SLOPE_GROUP_N_JOBS,
        skip_existing=skip_existing,
    )
    for path in out_paths:
        print(f"Wrote group {path}")
    return out_paths


def export_group_mean_slope_images(group_stats_file: Path, desc: str) -> list[Path]:
    result = load_regression_group_result(group_stats_file)
    out_paths: list[Path] = []

    for roi_name in EXPORT_ROIS:
        roi_idx = _find_roi_index(result.region_names, roi_name)
        resolved_roi = str(result.region_names[roi_idx])
        output_path = (
            Path(EXPORT_OUTPUT_DIR)
            / f"group_mean_slope_{_safe_filename_part(resolved_roi)}_"
            f"{_safe_filename_part(desc)}.{EXPORT_OUTPUT_FORMAT}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = _draw_mean_slope_figure(result, roi_idx)
        fig.savefig(
            output_path,
            dpi=EXPORT_OUTPUT_DPI,
            bbox_inches="tight",
            transparent=EXPORT_TRANSPARENT,
            format=EXPORT_OUTPUT_FORMAT,
        )
        fig.clf()
        out_paths.append(output_path)
        print(f"Exported {resolved_roi}: {output_path}")

    return out_paths


# ---------------------------------------------------------------------------
# Subject-level correction
# ---------------------------------------------------------------------------


def find_subject_regression_files(search_root: Path, desc: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(Path(search_root).glob(f"**/*desc-{desc}_stats.h5")):
        if path.is_file() and is_subject_regression_hdf5(path):
            paths.append(path)
    return paths


def is_subject_regression_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as fh:
            return (
                "regression" in fh
                and h5_string(fh, "meta/analysis_type") == "slope_regression"
            )
    except OSError:
        return False


def path_with_replaced_desc(path: Path, source_desc: str, target_desc: str) -> Path:
    source_token = f"desc-{source_desc}"
    target_token = f"desc-{target_desc}"
    if source_token not in path.name:
        raise ValueError(f"{path.name}: missing {source_token!r}.")
    return path.with_name(path.name.replace(source_token, target_token, 1))


def correct_subject_hdf5(path: Path, *, source_desc: str, target_desc: str) -> None:
    with h5py.File(path, "r+") as fh:
        multiply_dataset_in_place(fh, "regression/condition_b/slope", -1.0)
        multiply_dataset_in_place(fh, "regression/condition_b/r_value", -1.0)
        multiply_dataset_in_place(fh, "regression/condition_b/permuted_slopes", -1.0)
        multiply_dataset_in_place(fh, "predictor/condition_b_transformed_values", -1.0)
        multiply_dataset_in_place(fh, "predictor/condition_b_values", -1.0)

        update_predictor_transform_metadata(fh)
        write_string_dataset(
            fh,
            "meta/posthoc_unpleasant_sign_correction",
            "true",
        )
        write_string_dataset(fh, "meta/posthoc_source_desc", source_desc)
        write_string_dataset(fh, "meta/posthoc_output_desc", target_desc)


def multiply_dataset_in_place(fh: h5py.File, key: str, factor: float) -> None:
    if key not in fh:
        return
    ds = fh[key]
    ds[...] = np.asarray(ds[...]) * factor


def update_predictor_transform_metadata(fh: h5py.File) -> None:
    key = "meta/predictor_transform_by_condition_json"
    raw = h5_string(fh, key) if key in fh else "{}"
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    unpleasant = payload.get("unpleasant")
    if not isinstance(unpleasant, dict):
        unpleasant = {}
    unpleasant["scale"] = 1.0
    unpleasant.setdefault("offset", 0.0)
    payload["unpleasant"] = unpleasant

    write_string_dataset(
        fh,
        key,
        json.dumps(payload, sort_keys=True, ensure_ascii=True),
    )


def write_string_dataset(fh: h5py.File, key: str, value: str) -> None:
    parent_key, name = key.rsplit("/", 1)
    parent = fh.require_group(parent_key)
    if name in parent:
        del parent[name]
    parent.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))


def h5_string(fh: h5py.File, key: str) -> str:
    if key not in fh:
        return ""
    value = fh[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# Trial audit TSV correction
# ---------------------------------------------------------------------------


def copy_and_correct_trial_tsv(source_h5: Path, target_h5: Path) -> Path | None:
    source_tsv = trial_tsv_path(source_h5)
    if not source_tsv.exists():
        return None

    target_tsv = trial_tsv_path(target_h5)
    target_tsv.parent.mkdir(parents=True, exist_ok=True)
    correct_trial_tsv(source_tsv, target_tsv)
    return target_tsv


def trial_tsv_path(stats_path: Path) -> Path:
    return stats_path.with_name(stats_path.name.replace("_stats.h5", "_trials.tsv"))


def correct_trial_tsv(source_tsv: Path, target_tsv: Path) -> None:
    with open(source_tsv, "r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{source_tsv}: missing TSV header.")
        rows = [correct_trial_row(row) for row in reader]
        fieldnames = list(reader.fieldnames)

    with open(target_tsv, "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def correct_trial_row(row: dict[str, str]) -> dict[str, str]:
    if row.get("resolved_label", "").strip().casefold() != "unpleasant":
        return dict(row)

    corrected = dict(row)
    for key in ("predictor_transformed_value", "predictor_value"):
        corrected[key] = flip_numeric_string(corrected.get(key, ""))
    if "predictor_transform_scale" in corrected:
        corrected["predictor_transform_scale"] = "1.0"
    return corrected


def flip_numeric_string(value: str) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(parsed):
        return value
    return f"{-parsed:.17g}"


if __name__ == "__main__":
    main()
