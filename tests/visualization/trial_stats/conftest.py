"""Shared fixtures for visualization/trial_stats tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import RegressionParams
from gin_bids_py_analysis.processing.trial_stats import RegressionProcessingResult
from gin_bids_py_analysis.processing.trial_stats import ConditionTestParams
from gin_bids_py_analysis.processing.trial_stats import ConditionTestProcessingResult


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
def default_params() -> ConditionTestParams:
    return ConditionTestParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        condition_a="accepted",
        condition_b="rejected",
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
    )


@pytest.fixture()
def synthetic_result(default_params: ConditionTestParams) -> ConditionTestProcessingResult:
    """ConditionTestProcessingResult filled with synthetic arrays (n_channels=4, n_times=60)."""
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

    return ConditionTestProcessingResult(
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


@pytest.fixture()
def default_slope_params() -> RegressionParams:
    return RegressionParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        condition_a="accepted",
        condition_b="rejected",
        predictor="predictor_value",
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
    )


@pytest.fixture()
def synthetic_slope_result(default_slope_params: RegressionParams) -> RegressionProcessingResult:
    rng = np.random.default_rng(seed=123)
    n_ch, n_t = 4, 60

    mock_file = _make_bids_file(
        Path("/fake/bids/sub-01_task-test_ieeg.vhdr"),
        {"subject": "01", "task": "test", "suffix": "ieeg"},
    )
    group = BIDSFileGroup(primary=mock_file)

    slope_a = rng.normal(loc=0.5, scale=0.2, size=(n_ch, n_t))
    slope_b = rng.normal(loc=-0.3, scale=0.2, size=(n_ch, n_t))
    p_a_raw = rng.uniform(0, 1, (n_ch, n_t))
    p_b_raw = rng.uniform(0, 1, (n_ch, n_t))
    p_a = np.clip(p_a_raw, 0, 1)
    p_b = np.clip(p_b_raw, 0, 1)
    sig_a = p_a < 0.05
    sig_b = p_b < 0.05
    mean_a = rng.standard_normal((n_ch, n_t))
    mean_b = rng.standard_normal((n_ch, n_t))
    epoch_means_a = rng.standard_normal((n_ch, 12))
    epoch_means_b = rng.standard_normal((n_ch, 11))

    return RegressionProcessingResult(
        source_group=group,
        metadata={},
        output_entities=None,
        condition_a_slope=slope_a,
        condition_a_intercept=rng.standard_normal((n_ch, n_t)),
        condition_a_r_value=np.clip(rng.standard_normal((n_ch, n_t)) * 0.3, -1, 1),
        condition_a_p_value=p_a_raw,
        condition_a_p_value_corrected=p_a,
        condition_a_significant_mask=sig_a,
        condition_b_slope=slope_b,
        condition_b_intercept=rng.standard_normal((n_ch, n_t)),
        condition_b_r_value=np.clip(rng.standard_normal((n_ch, n_t)) * 0.3, -1, 1),
        condition_b_p_value=p_b_raw,
        condition_b_p_value_corrected=p_b,
        condition_b_significant_mask=sig_b,
        condition_a_mean=mean_a,
        condition_b_mean=mean_b,
        condition_a_sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
        condition_b_sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
        time_axis_s=np.linspace(-1.0, 2.0, n_t),
        channel_names=["A1", "A2", "A3", "A4"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=12,
        condition_b_trial_count=11,
        condition_a_trials_used=12,
        condition_b_trials_used=11,
        sfreq=20.0,
        condition_a_predictor_values=np.linspace(0.0, 1.0, 12),
        condition_b_predictor_values=np.linspace(0.0, 1.0, 11),
        condition_a_trial_activity_summary_values=epoch_means_a.copy(),
        condition_b_trial_activity_summary_values=epoch_means_b.copy(),
        condition_a_epoch_means=epoch_means_a,
        condition_b_epoch_means=epoch_means_b,
        resolved_trials=[],
        source_ieeg_files=[str(mock_file.path)],
        source_table_files=[],
        source_electrodes_files=[],
        analysis_level="channel",
        analysis_type="slope_regression",
        predictor="predictor_value",
        predictor_zscore="none",
        trial_activity_summary_kind="epoch_mean",
        trial_activity_summary_missing_response_policy="clamp_to_epoch",
        trial_activity_summary_source={},
        trial_activity_summary_label="Epoch mean activity",
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        condition_a_stats_valid=True,
        condition_b_stats_valid=True,
        stats_valid=True,
    )


