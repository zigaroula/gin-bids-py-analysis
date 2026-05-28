"""Shared fixtures for visualization/trial_stats tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.trial_stats import RegressionParams
from bidsforge.processing.trial_stats import RegressionProcessingResult
from bidsforge.processing.trial_stats.regression import (
    ConditionPredictorValues,
    ConditionRegressionStats,
    RegressionPredictor,
    RegressionStats,
)
from bidsforge.processing.trial_stats import ConditionTestParams
from bidsforge.processing.trial_stats import ConditionTestProcessingResult
from bidsforge.processing.trial_stats import ConditionContrast, DifferenceEstimate
from bidsforge.processing.trial_stats import (
    SignalActivityEstimate,
    ConditionSignalActivity,
    ConditionTrialSummaryValues,
)


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
        signal_activity=ConditionSignalActivity(
            condition_a=SignalActivityEstimate(
                mean=mean_a,
                sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
            ),
            condition_b=SignalActivityEstimate(
                mean=mean_b,
                sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
            ),
        ),
        difference=DifferenceEstimate(
            mean=mean_a - mean_b,
            sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
            ci95_low=mean_a - mean_b - 0.2,
            ci95_high=mean_a - mean_b + 0.2,
        ),
        contrast=ConditionContrast(
            t_values=t_vals,
            p_values=p_corr,
            p_values_uncorrected=p_raw,
            significant_mask=sig_mask,
        ),
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
        regression=RegressionStats(
            condition_a=ConditionRegressionStats(
                slope=slope_a,
                intercept=rng.standard_normal((n_ch, n_t)),
                r_value=np.clip(rng.standard_normal((n_ch, n_t)) * 0.3, -1, 1),
                p_value=p_a_raw,
                p_value_corrected=p_a,
                significant_mask=sig_a,
                n_trials_used=12,
                stats_valid=True,
            ),
            condition_b=ConditionRegressionStats(
                slope=slope_b,
                intercept=rng.standard_normal((n_ch, n_t)),
                r_value=np.clip(rng.standard_normal((n_ch, n_t)) * 0.3, -1, 1),
                p_value=p_b_raw,
                p_value_corrected=p_b,
                significant_mask=sig_b,
                n_trials_used=11,
                stats_valid=True,
            ),
        ),
        predictor_values=RegressionPredictor(
            condition_a=ConditionPredictorValues(values=np.linspace(0.0, 1.0, 12)),
            condition_b=ConditionPredictorValues(values=np.linspace(0.0, 1.0, 11)),
        ),
        signal_activity=ConditionSignalActivity(
            condition_a=SignalActivityEstimate(
                mean=mean_a,
                sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
            ),
            condition_b=SignalActivityEstimate(
                mean=mean_b,
                sem=np.abs(rng.standard_normal((n_ch, n_t))) * 0.1,
            ),
        ),
        time_axis_s=np.linspace(-1.0, 2.0, n_t),
        channel_names=["A1", "A2", "A3", "A4"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=12,
        condition_b_trial_count=11,
        sfreq=20.0,
        trial_activity_summary_values=ConditionTrialSummaryValues(
            condition_a=epoch_means_a.copy(),
            condition_b=epoch_means_b.copy(),
        ),
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
        stats_valid=True,
    )





