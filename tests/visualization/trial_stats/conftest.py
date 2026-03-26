"""Shared fixtures for visualization/trial_stats tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import TrialStatsParams
from gin_bids_py_analysis.processing.trial_stats.result import TrialStatsProcessingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_params() -> TrialStatsParams:
    return TrialStatsParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        condition_a="accepted",
        condition_b="rejected",
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
    )


@pytest.fixture()
def synthetic_result(default_params: TrialStatsParams) -> TrialStatsProcessingResult:
    """TrialStatsProcessingResult filled with synthetic arrays (n_channels=4, n_times=60)."""
    rng = np.random.default_rng(seed=42)
    n_ch, n_t = 4, 60

    mock_file = _make_bids_file(
        Path("/fake/bids/sub-01_task-test_ieeg.vhdr"),
        {"subject": "01", "task": "test", "suffix": "ieeg"},
    )
    group = BIDSFileGroup(primary=mock_file)

    t_vals = rng.standard_normal((n_ch, n_t))
    p_raw = rng.uniform(0, 1, (n_ch, n_t))
    p_corr = np.clip(p_raw, 0, 1)
    sig_mask = p_corr < 0.05
    mean_a = rng.standard_normal((n_ch, n_t))
    mean_b = rng.standard_normal((n_ch, n_t))

    return TrialStatsProcessingResult(
        source_group=group,
        metadata={},
        output_entities=None,
        t_values=t_vals,
        p_values=p_corr,
        p_values_uncorrected=p_raw,
        significant_mask=sig_mask,
        condition_a_mean=mean_a,
        condition_b_mean=mean_b,
        mean_difference=mean_a - mean_b,
        condition_a_sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
        condition_b_sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
        difference_sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
        difference_ci95_low=mean_a - mean_b - 0.2,
        difference_ci95_high=mean_a - mean_b + 0.2,
        time_axis_s=np.linspace(-1.0, 2.0, n_t),
        channel_names=["A1", "A2", "A3", "A4"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=10,
        condition_b_trial_count=8,
        sfreq=20.0,
        resolved_trials=[],
        source_ieeg_files=[str(mock_file.path)],
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
    )
