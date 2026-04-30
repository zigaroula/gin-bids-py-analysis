from __future__ import annotations

import csv
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module() -> object:
    script_path = REPO_ROOT / "scripts" / "correct_trial_slope_unpleasant_sign.py"
    spec = spec_from_file_location("correct_trial_slope_unpleasant_sign", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_subject_h5(path: Path, *, desc: str) -> None:
    del desc
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as fh:
        meta = fh.create_group("meta")
        meta.create_dataset("analysis_type", data="slope_regression", dtype=str_dtype)
        meta.create_dataset(
            "predictor_transform_by_condition_json",
            data=json.dumps(
                {
                    "pleasant": {"scale": 1.0, "offset": 0.0},
                    "unpleasant": {"scale": -1.0, "offset": 0.0},
                },
                sort_keys=True,
            ),
            dtype=str_dtype,
        )
        regression = fh.create_group("regression")
        cond_a = regression.create_group("condition_a")
        cond_b = regression.create_group("condition_b")
        cond_a.create_dataset("slope", data=np.array([[1.0, 2.0]]))
        cond_a.create_dataset("r_value", data=np.array([[0.1, 0.2]]))
        cond_a.create_dataset("p_value", data=np.array([[0.01, 0.02]]))
        cond_b.create_dataset("slope", data=np.array([[3.0, -4.0]]))
        cond_b.create_dataset("r_value", data=np.array([[0.3, -0.4]]))
        cond_b.create_dataset("p_value", data=np.array([[0.03, 0.04]]))
        cond_b.create_dataset("permuted_slopes", data=np.array([[[5.0, -6.0]]]))
        predictor = fh.create_group("predictor")
        predictor.create_dataset("condition_a_transformed_values", data=np.array([1.0, 2.0]))
        predictor.create_dataset("condition_b_transformed_values", data=np.array([-3.0, 4.0]))
        predictor.create_dataset("condition_a_values", data=np.array([1.0, 2.0]))
        predictor.create_dataset("condition_b_values", data=np.array([-3.0, 4.0]))


def _write_trials_tsv(path: Path) -> None:
    tsv_path = path.with_name(path.name.replace("_stats.h5", "_trials.tsv"))
    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "resolved_label",
                "predictor_transformed_value",
                "predictor_value",
                "predictor_transform_scale",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "resolved_label": "pleasant",
                "predictor_transformed_value": "2.5",
                "predictor_value": "2.5",
                "predictor_transform_scale": "1.0",
            }
        )
        writer.writerow(
            {
                "resolved_label": "unpleasant",
                "predictor_transformed_value": "-3.5",
                "predictor_value": "-3.5",
                "predictor_transform_scale": "-1.0",
            }
        )


def test_correct_subject_files_discovers_three_descs_and_flips_condition_b(tmp_path) -> None:
    module = _load_script_module()
    root = tmp_path / "derivatives" / "regression"

    for source_desc in module.DESCRIPTION_MAP:
        source = (
            root
            / f"sub-01/ieeg/sub-01_task-test_desc-{source_desc}_stats.h5"
        )
        _write_subject_h5(source, desc=source_desc)
        _write_trials_tsv(source)

    for source_desc, target_desc in module.DESCRIPTION_MAP.items():
        out_paths = module.correct_subject_files(
            source_desc=source_desc,
            target_desc=target_desc,
            search_root=root,
            skip_existing=False,
            dry_run=False,
        )
        assert len(out_paths) == 1
        target = out_paths[0]
        assert f"desc-{target_desc}" in target.name

        with h5py.File(target, "r") as fh:
            np.testing.assert_allclose(fh["regression/condition_a/slope"][:], [[1.0, 2.0]])
            np.testing.assert_allclose(fh["regression/condition_a/r_value"][:], [[0.1, 0.2]])
            np.testing.assert_allclose(fh["regression/condition_a/p_value"][:], [[0.01, 0.02]])
            np.testing.assert_allclose(fh["regression/condition_b/slope"][:], [[-3.0, 4.0]])
            np.testing.assert_allclose(fh["regression/condition_b/r_value"][:], [[-0.3, 0.4]])
            np.testing.assert_allclose(fh["regression/condition_b/p_value"][:], [[0.03, 0.04]])
            np.testing.assert_allclose(fh["regression/condition_b/permuted_slopes"][:], [[[-5.0, 6.0]]])
            np.testing.assert_allclose(fh["predictor/condition_b_transformed_values"][:], [3.0, -4.0])
            np.testing.assert_allclose(fh["predictor/condition_b_values"][:], [3.0, -4.0])
            payload = json.loads(module.h5_string(fh, "meta/predictor_transform_by_condition_json"))
            assert payload["unpleasant"]["scale"] == 1.0
            assert module.h5_string(fh, "meta/posthoc_unpleasant_sign_correction") == "true"
            assert module.h5_string(fh, "meta/posthoc_source_desc") == source_desc

        target_tsv = target.with_name(target.name.replace("_stats.h5", "_trials.tsv"))
        with open(target_tsv, "r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert rows[0]["resolved_label"] == "pleasant"
        assert rows[0]["predictor_value"] == "2.5"
        assert rows[1]["resolved_label"] == "unpleasant"
        assert rows[1]["predictor_transformed_value"] == "3.5"
        assert rows[1]["predictor_value"] == "3.5"
        assert rows[1]["predictor_transform_scale"] == "1.0"


def test_correct_subject_files_skip_existing_does_not_overwrite(tmp_path) -> None:
    module = _load_script_module()
    root = tmp_path / "derivatives" / "regression"
    source = root / "sub-01/ieeg/sub-01_task-test_desc-onset_stats.h5"
    _write_subject_h5(source, desc="onset")

    target = source.with_name("sub-01_task-test_desc-onsetnoflip_stats.h5")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    with h5py.File(target, "r+") as fh:
        fh["regression/condition_b/slope"][...] = np.array([[99.0, 100.0]])

    out_paths = module.correct_subject_files(
        source_desc="onset",
        target_desc="onsetnoflip",
        search_root=root,
        skip_existing=True,
        dry_run=False,
    )

    assert out_paths == [target]
    with h5py.File(target, "r") as fh:
        np.testing.assert_allclose(fh["regression/condition_b/slope"][:], [[99.0, 100.0]])


def test_export_group_mean_slope_images_for_vmPFC_and_aIns(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    group_path = tmp_path / "sub-group_task-test_desc-onsetnoflip_stats.h5"
    group_path.touch()

    fake_result = SimpleNamespace(
        region_names=["vmPFC", "aIns"],
        time_axis_s=np.array([0.0, 1.0]),
        condition_labels=("pleasant", "unpleasant"),
        source_metric="slope",
        roi_channel_counts=np.array([2, 3]),
        roi_subject_counts=np.array([1, 2]),
        condition_a_source_metric_mean=np.array([[1.0, 2.0], [3.0, 4.0]]),
        condition_a_source_metric_sem=np.zeros((2, 2)),
        condition_b_source_metric_mean=np.array([[-1.0, -2.0], [-3.0, -4.0]]),
        condition_b_source_metric_sem=np.zeros((2, 2)),
        condition_a_source_metric_vs_zero_significant_mask=np.zeros((2, 2), dtype=bool),
        condition_b_source_metric_vs_zero_significant_mask=np.zeros((2, 2), dtype=bool),
        source_metric_significant_mask=np.zeros((2, 2), dtype=bool),
    )
    monkeypatch.setattr(module, "load_regression_group_result", lambda _path: fake_result)
    monkeypatch.setattr(module, "EXPORT_OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "EXPORT_OUTPUT_FORMAT", "png")

    out_paths = module.export_group_mean_slope_images(group_path, "onsetnoflip")

    assert [path.name for path in out_paths] == [
        "group_mean_slope_vmPFC_onsetnoflip.png",
        "group_mean_slope_aIns_onsetnoflip.png",
    ]
    assert all(path.exists() for path in out_paths)
