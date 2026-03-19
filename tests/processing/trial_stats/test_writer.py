from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    ResolvedTrial,
    TrialStatsProcessingResult,
    TrialStatsProcessingWriter,
    TrialStatsWriterParams,
)


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


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
        assert list(fh["axes"]["channel"].asstr()[:]) == ["A1", "A2"]
        assert list(fh["meta"]["trial_counts"][:]) == [4, 4]
        assert fh["meta"]["p_value_correction_method"].asstr()[()] == "fdr_bh"
        assert float(fh["meta"]["significance_alpha"][()]) == 0.05
        assert fh["meta"]["analysis_level"].asstr()[()] == "roi"
        assert fh["meta"]["atlas_name"].asstr()[()] == "atlasA"
        assert list(fh["meta"]["atlas_regions"].asstr()[:]) == ["R1", "R2"]
        assert float(fh["meta"]["window_ms"][()]) == 100.0
        assert int(fh["meta"]["n_bins"][()]) == 0
        assert int(fh["meta"]["effective_n_bins"][()]) == 3
        assert fh["meta"]["binning_mode"].asstr()[()] == "window_ms"
        assert list(fh["provenance"]["source_ieeg_files"].asstr()[:]) == [str(primary.path)]
        assert list(fh["provenance"]["source_electrodes_files"].asstr()[:]) == [str(tmp_path / "electrodes.tsv")]

    lines = trial_table_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("source_file\tanchor_event_index")
    assert "accepted" in lines[1]

