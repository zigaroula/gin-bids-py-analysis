from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    ResolvedTrial,
    TrialStatsProcessingResult,
    TrialStatsProcessingWriter,
    TrialStatsWriterParams,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_case_dir(case_name: str) -> Path:
    root = REPO_ROOT / "tests" / "_script_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{case_name}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def test_writer_outputs_hdf5_and_trial_table_with_grouped_entities(
    tmp_path: Path,
) -> None:
    primary = _make_bids_file(
        tmp_path / "sub-01_ses-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    result = TrialStatsProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        output_entities={"subject": "01", "task": "decid"},
        t_values=np.ones((2, 3), dtype=np.float64),
        p_values=np.full((2, 3), 0.05, dtype=np.float64),
        p_values_uncorrected=np.full((2, 3), 0.01, dtype=np.float64),
        condition_a_mean=np.full((2, 3), 5.0, dtype=np.float64),
        condition_b_mean=np.full((2, 3), 1.0, dtype=np.float64),
        mean_difference=np.full((2, 3), 4.0, dtype=np.float64),
        condition_a_sem=np.full((2, 3), 0.5, dtype=np.float64),
        condition_b_sem=np.full((2, 3), 0.4, dtype=np.float64),
        difference_sem=np.full((2, 3), 0.64, dtype=np.float64),
        difference_ci95_low=np.full((2, 3), 2.7, dtype=np.float64),
        difference_ci95_high=np.full((2, 3), 5.3, dtype=np.float64),
        significant_mask=np.ones((2, 3), dtype=bool),
        time_axis_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        channel_names=["A1", "A2"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=4,
        condition_b_trial_count=4,
        sfreq=10.0,
        resolved_trials=[
            ResolvedTrial(
                source_file=primary,
                anchor_event_index=0,
                anchor_event_code="10",
                anchor_onset_s=1.0,
                anchor_duration_s=0.0,
                label="accepted",
                keep=True,
            )
        ],
        source_ieeg_files=[str(primary.path)],
        source_table_files=[str(tmp_path / "labels.tsv")],
        source_electrodes_files=[str(tmp_path / "electrodes.tsv")],
        analysis_level="roi",
        atlas_name="atlasA",
        atlas_regions=["R1", "R2"],
        region_channels={
            "A1": ["CH01", "CH02"],
            "A2": ["CH03"],
        },
        window_ms=100.0,
        n_bins=0,
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        stats_valid=True,
    )
    writer = TrialStatsProcessingWriter(
        TrialStatsWriterParams(
            bids_root=tmp_path,
            output_description="acceptreject",
        )
    )

    output_path = writer.write(result)
    trial_table_path = output_path.with_suffix(".tsv").with_name(
        output_path.with_suffix(".tsv").name.replace("_stats.tsv", "_trials.tsv")
    )

    assert output_path.exists()
    assert trial_table_path.exists()
    assert "run-" not in output_path.name
    assert "ses-" not in str(output_path.parent)

    with h5py.File(output_path, "r") as fh:
        assert fh["stats"]["t_values"].shape == (2, 3)
        assert fh["stats"]["p_values_uncorrected"].shape == (2, 3)
        assert fh["stats"]["significant_mask"].shape == (2, 3)
        assert fh["means"]["difference"].shape == (2, 3)
        assert fh["uncertainty"]["accepted_sem"].shape == (2, 3)
        assert fh["uncertainty"]["rejected_sem"].shape == (2, 3)
        assert fh["uncertainty"]["difference_sem"].shape == (2, 3)
        assert fh["uncertainty"]["difference_ci95_low"].shape == (2, 3)
        assert fh["uncertainty"]["difference_ci95_high"].shape == (2, 3)
        assert list(fh["axes"]["region"].asstr()[:]) == ["A1", "A2"]
        assert "channel" not in fh["axes"]
        assert list(fh["meta"]["trial_counts"][:]) == [4, 4]
        assert fh["meta"]["p_value_correction_method"].asstr()[()] == "fdr_bh"
        assert float(fh["meta"]["significance_alpha"][()]) == 0.05
        assert fh["meta"]["analysis_level"].asstr()[()] == "roi"
        assert fh["meta"]["atlas_name"].asstr()[()] == "atlasA"
        assert list(fh["meta"]["atlas_regions"].asstr()[:]) == ["R1", "R2"]
        assert list(fh["meta"]["atlas_region_channel_map"]["region_order"].asstr()[:]) == ["A1", "A2"]
        assert list(fh["meta"]["atlas_region_channel_map"]["region"].asstr()[:]) == ["A1", "A1", "A2"]
        assert list(fh["meta"]["atlas_region_channel_map"]["channel"].asstr()[:]) == ["CH01", "CH02", "CH03"]
        assert float(fh["meta"]["window_ms"][()]) == 100.0
        assert int(fh["meta"]["n_bins"][()]) == 0
        assert int(fh["meta"]["effective_n_bins"][()]) == 3
        assert fh["meta"]["binning_mode"].asstr()[()] == "window_ms"
        assert list(fh["provenance"]["source_ieeg_files"].asstr()[:]) == [str(primary.path)]
        assert list(fh["provenance"]["source_electrodes_files"].asstr()[:]) == [str(tmp_path / "electrodes.tsv")]

    lines = trial_table_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("source_file\tanchor_event_index")
    assert "accepted" in lines[1]


def test_writer_outputs_matlab_and_trial_table() -> None:
    case_dir = _make_case_dir("writer_ts_matlab")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_ieeg.vhdr",
            {
                "subject": "01",
                "task": "decid",
                "suffix": "ieeg",
                "extension": ".vhdr",
                "datatype": "ieeg",
            },
        )
        result = TrialStatsProcessingResult(
            source_group=BIDSFileGroup(primary=primary),
            output_entities={"subject": "01", "task": "decid"},
            t_values=np.ones((2, 3), dtype=np.float64),
            p_values=np.full((2, 3), 0.04, dtype=np.float64),
            p_values_uncorrected=np.full((2, 3), 0.01, dtype=np.float64),
            condition_a_mean=np.full((2, 3), 5.0, dtype=np.float64),
            condition_b_mean=np.full((2, 3), 1.0, dtype=np.float64),
            mean_difference=np.full((2, 3), 4.0, dtype=np.float64),
            condition_a_sem=np.full((2, 3), 0.5, dtype=np.float64),
            condition_b_sem=np.full((2, 3), 0.4, dtype=np.float64),
            difference_sem=np.full((2, 3), 0.64, dtype=np.float64),
            difference_ci95_low=np.full((2, 3), 2.7, dtype=np.float64),
            difference_ci95_high=np.full((2, 3), 5.3, dtype=np.float64),
            significant_mask=np.ones((2, 3), dtype=bool),
            time_axis_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            channel_names=["A1", "A2"],
            condition_a="accepted",
            condition_b="rejected",
            condition_a_trial_count=3,
            condition_b_trial_count=3,
            sfreq=10.0,
            resolved_trials=[
                ResolvedTrial(
                    source_file=primary,
                    anchor_event_index=0,
                    anchor_event_code="10",
                    anchor_onset_s=1.0,
                    anchor_duration_s=0.0,
                    label="accepted",
                    keep=True,
                )
            ],
            source_ieeg_files=[str(primary.path)],
            source_table_files=[str(case_dir / "labels.tsv")],
            source_electrodes_files=[str(case_dir / "electrodes.tsv")],
            analysis_level="channel",
            atlas_name=None,
            atlas_regions=[],
            region_channels={},
            window_ms=0.0,
            n_bins=0,
            p_value_correction_method="fdr_bh",
            significance_alpha=0.05,
            stats_valid=True,
        )
        writer = TrialStatsProcessingWriter(
            TrialStatsWriterParams(
                bids_root=case_dir,
                output_description="acceptreject",
                output_format="matlab",
            )
        )

        output_path = writer.write(result)
        trial_table_path = output_path.with_suffix(".tsv").with_name(
            output_path.with_suffix(".tsv").name.replace("_stats.tsv", "_trials.tsv")
        )

        assert output_path.suffix == ".mat"
        assert output_path.exists()
        assert trial_table_path.exists()

        mat = scipy.io.loadmat(str(output_path), squeeze_me=True, struct_as_record=False)
        data = mat["data"]
        # With squeeze_me=True, shapes are exactly as stored (no singleton dims here)
        assert data.stats.t_values.shape == (2, 3)
        assert data.stats.p_values.shape == (2, 3)
        assert data.stats.p_values_uncorrected.shape == (2, 3)
        assert data.stats.significant_mask.shape == (2, 3)
        assert data.means.difference.shape == (2, 3)
        assert data.means.accepted.shape == (2, 3)
        assert data.means.rejected.shape == (2, 3)
        assert data.uncertainty.accepted_sem.shape == (2, 3)
        assert data.uncertainty.rejected_sem.shape == (2, 3)
        assert data.uncertainty.difference_sem.shape == (2, 3)
        assert data.uncertainty.difference_ci95_low.shape == (2, 3)
        assert data.uncertainty.difference_ci95_high.shape == (2, 3)
        # channel axis: object array of 2 names
        channel_arr = np.atleast_1d(data.axes.channel)
        assert channel_arr.shape[0] == 2
        assert list(channel_arr) == ["A1", "A2"]
        np.testing.assert_allclose(data.axes.time_s, [0.0, 0.1, 0.2])
        assert str(data.provenance.pipeline_name) == "trialstats"
        # trial counts array: 2-element after squeeze
        counts = np.atleast_1d(data.meta.trial_counts)
        assert int(counts[0]) == 3

        lines = trial_table_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("source_file\tanchor_event_index")
        assert "accepted" in lines[1]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def _make_minimal_result(
    tmp_path: Path,
    channel_significant_mask=None,
) -> "TrialStatsProcessingResult":
    """Build the smallest valid TrialStatsProcessingResult for writer tests."""
    primary = _make_bids_file(
        tmp_path / "sub-01_task-t_ieeg.vhdr",
        {"subject": "01", "task": "t", "suffix": "ieeg", "extension": ".vhdr"},
    )
    return TrialStatsProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        output_entities={"subject": "01", "task": "t"},
        t_values=np.ones((2, 3), dtype=np.float64),
        p_values=np.full((2, 3), 0.04, dtype=np.float64),
        p_values_uncorrected=np.full((2, 3), 0.01, dtype=np.float64),
        condition_a_mean=np.full((2, 3), 5.0, dtype=np.float64),
        condition_b_mean=np.full((2, 3), 1.0, dtype=np.float64),
        mean_difference=np.full((2, 3), 4.0, dtype=np.float64),
        condition_a_sem=np.full((2, 3), 0.5, dtype=np.float64),
        condition_b_sem=np.full((2, 3), 0.4, dtype=np.float64),
        difference_sem=np.full((2, 3), 0.64, dtype=np.float64),
        difference_ci95_low=np.full((2, 3), 2.7, dtype=np.float64),
        difference_ci95_high=np.full((2, 3), 5.3, dtype=np.float64),
        significant_mask=np.ones((2, 3), dtype=bool),
        time_axis_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        channel_names=["A1", "A2"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=2,
        condition_b_trial_count=2,
        sfreq=10.0,
        resolved_trials=[
            ResolvedTrial(
                source_file=primary,
                anchor_event_index=0,
                anchor_event_code="10",
                anchor_onset_s=1.0,
                anchor_duration_s=0.0,
                label="accepted",
                keep=True,
            )
        ],
        source_ieeg_files=[str(primary.path)],
        source_table_files=[],
        source_electrodes_files=[],
        analysis_level="channel",
        atlas_name=None,
        atlas_regions=[],
        region_channels={},
        window_ms=0.0,
        n_bins=0,
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        stats_valid=True,
        channel_significant_mask=channel_significant_mask,
        metadata={
            "channel_significance_mode": "none" if channel_significant_mask is None else "single_bin",
            "channel_significance_duration_threshold_ms": 100.0,
        },
    )


def test_writer_hdf5_channel_significant_mask_written_and_loaded(
    tmp_path: Path,
) -> None:
    from gin_bids_py_analysis.processing.trial_stats.result_loader import load_trial_stats_result

    mask = np.array([True, False], dtype=bool)
    result = _make_minimal_result(tmp_path, channel_significant_mask=mask)

    writer = TrialStatsProcessingWriter(
        TrialStatsWriterParams(bids_root=tmp_path)
    )
    output_path = writer.write(result)

    # Verify dataset is in the HDF5 file with correct content
    with h5py.File(output_path, "r") as fh:
        assert "channel_significant_mask" in fh["stats"]
        stored = np.asarray(fh["stats"]["channel_significant_mask"][:], dtype=bool)
        np.testing.assert_array_equal(stored, mask)
        assert fh["meta"]["channel_significance_mode"].asstr()[()] == "single_bin"
        assert float(fh["meta"]["channel_significance_duration_threshold_ms"][()]) == 100.0

    # Round-trip through the loader
    loaded = load_trial_stats_result(output_path)
    assert loaded.channel_significant_mask is not None
    np.testing.assert_array_equal(loaded.channel_significant_mask, mask)


def test_writer_hdf5_channel_significant_mask_absent_when_none(
    tmp_path: Path,
) -> None:
    from gin_bids_py_analysis.processing.trial_stats.result_loader import load_trial_stats_result

    result = _make_minimal_result(tmp_path, channel_significant_mask=None)

    writer = TrialStatsProcessingWriter(
        TrialStatsWriterParams(bids_root=tmp_path)
    )
    output_path = writer.write(result)

    with h5py.File(output_path, "r") as fh:
        assert "channel_significant_mask" not in fh["stats"]

    loaded = load_trial_stats_result(output_path)
    assert loaded.channel_significant_mask is None

