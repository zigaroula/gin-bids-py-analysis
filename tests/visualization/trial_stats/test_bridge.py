"""Tests for the in-memory HDF5 bridge used by the Group tab."""

from __future__ import annotations

import numpy as np
import pytest

from bidsforge.visualization.trial_stats._bridge import (
    _write_hdf5_structure,
    build_group_file_group_from_results,
    build_in_memory_bids_file,
)


class TestWriteHdf5Structure:
    """Verify that _write_hdf5_structure produces the correct schema."""

    def test_all_required_datasets_present(self, synthetic_result):
        import h5py

        with h5py.File("test.h5", "w", driver="core", backing_store=False) as fh:
            _write_hdf5_structure(fh, synthetic_result, "01")

            assert "axes/channel" in fh
            assert "axes/time_s" in fh
            assert "meta/analysis_level" in fh
            assert "meta/condition_labels" in fh
            assert "meta/binning_mode" in fh
            assert "meta/window_ms" in fh
            assert "meta/n_bins" in fh
            assert "meta/effective_n_bins" in fh
            assert "stats/condition_contrast/t_values" in fh
            assert "data/signal_activity/difference/mean" in fh
            assert f"data/signal_activity/{synthetic_result.condition_a}/mean" in fh
            assert f"data/signal_activity/{synthetic_result.condition_b}/mean" in fh
            assert "provenance/source_ieeg_files" in fh
            assert "provenance/source_electrodes_files" in fh

    def test_channel_names_round_trip(self, synthetic_result):
        import h5py

        with h5py.File("test.h5", "w", driver="core", backing_store=False) as fh:
            _write_hdf5_structure(fh, synthetic_result, "01")

            stored = [c.decode() if isinstance(c, bytes) else c for c in fh["axes/channel"][:]]
            assert stored == list(synthetic_result.channel_names)

    def test_t_values_shape(self, synthetic_result):
        import h5py

        with h5py.File("test.h5", "w", driver="core", backing_store=False) as fh:
            _write_hdf5_structure(fh, synthetic_result, "01")

            expected_shape = (len(synthetic_result.channel_names), len(synthetic_result.time_axis_s))
            assert fh["stats/condition_contrast/t_values"].shape == expected_shape

    def test_time_axis_values(self, synthetic_result):
        import h5py

        with h5py.File("test.h5", "w", driver="core", backing_store=False) as fh:
            _write_hdf5_structure(fh, synthetic_result, "01")

            np.testing.assert_allclose(fh["axes/time_s"][:], synthetic_result.time_axis_s)


class TestBuildInMemoryBIDSFile:
    """Verify that build_in_memory_bids_file returns a usable BIDSFile."""

    def test_extension_is_h5(self, synthetic_result):
        bids_file = build_in_memory_bids_file(synthetic_result, "01")
        assert bids_file.extension == ".h5"

    def test_subject_entity(self, synthetic_result):
        bids_file = build_in_memory_bids_file(synthetic_result, "42")
        assert bids_file.get("subject") == "42"

    def test_is_loaded(self, synthetic_result):
        bids_file = build_in_memory_bids_file(synthetic_result, "01")
        assert bids_file.is_loaded

    def test_ensure_loaded_yields_h5_handle(self, synthetic_result):
        import h5py

        bids_file = build_in_memory_bids_file(synthetic_result, "01")
        with bids_file.ensure_loaded() as fh:
            assert isinstance(fh, h5py.File)
            assert "axes/channel" in fh


class TestBuildGroupFileGroupFromResults:
    """Verify that build_group_file_group_from_results builds a valid BIDSFileGroup."""

    def test_returns_file_group_with_all_subjects(self, synthetic_result):
        results = {
            "01": synthetic_result,
            "02": synthetic_result,
        }
        group = build_group_file_group_from_results(results, primary_condition_metric="t_values")
        subject_ids = {f.get("subject") for f in group.all_files}
        assert "01" in subject_ids
        assert "02" in subject_ids

    def test_raises_on_empty_results(self):
        with pytest.raises(ValueError, match="at least one result"):
            build_group_file_group_from_results({}, primary_condition_metric="t_values")

    def test_single_subject(self, synthetic_result):
        group = build_group_file_group_from_results(
            {"01": synthetic_result}, primary_condition_metric="t_values"
        )
        assert len(group.all_files) == 1



