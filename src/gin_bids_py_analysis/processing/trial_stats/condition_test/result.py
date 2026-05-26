from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.utils.serialization import OutputTree, compressed

from ..result import BaseTrialStatsProcessingResult, _deep_merge


@dataclass
class DifferenceEstimate:
    """Mean and uncertainty for the condition_a − condition_b difference."""

    mean: np.ndarray
    sem: np.ndarray
    ci95_low: np.ndarray
    ci95_high: np.ndarray


@dataclass
class ConditionContrast:
    """Per-channel, per-time outputs from the condition A vs B statistical test."""

    t_values: np.ndarray
    p_values: np.ndarray
    p_values_uncorrected: np.ndarray
    significant_mask: np.ndarray
    permuted_t_values: np.ndarray | None = None
    channel_significant_mask: np.ndarray | None = None

    def to_output_dict(self, *, significance_alpha: float = 0.05) -> dict[str, object]:
        """Build the serializable dict for the condition_contrast stats block."""
        p_values_uncorrected = (
            self.p_values_uncorrected
            if self.p_values_uncorrected.size
            else self.p_values
        )
        significant_mask = (
            self.significant_mask
            if self.significant_mask.size
            else (np.isfinite(self.p_values) & (self.p_values < significance_alpha))
        )
        out: dict[str, object] = {
            "t_values": self.t_values.astype(np.float64),
            "p_values": self.p_values.astype(np.float64),
            "p_values_uncorrected": p_values_uncorrected.astype(np.float64),
            "significant_mask": significant_mask.astype(bool),
        }
        if self.channel_significant_mask is not None:
            out["channel_significant_mask"] = self.channel_significant_mask.astype(bool)
        if self.permuted_t_values is not None:
            out["permuted_t_values"] = compressed(
                self.permuted_t_values.astype(np.float32)
            )
        return out


@dataclass
class ConditionTestProcessingResult(BaseTrialStatsProcessingResult):
    """Structured outputs for subject-level condition-test statistics."""

    difference: DifferenceEstimate = field(
        default_factory=lambda: DifferenceEstimate(
            mean=np.array([]),
            sem=np.array([]),
            ci95_low=np.array([]),
            ci95_high=np.array([]),
        )
    )
    contrast: ConditionContrast = field(
        default_factory=lambda: ConditionContrast(
            t_values=np.array([]),
            p_values=np.array([]),
            p_values_uncorrected=np.array([]),
            significant_mask=np.array([]),
        )
    )

    def to_output_tree(
        self,
        *,
        include_epochs: bool = False,
        pipeline_name: str = "unknown",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return the full output tree for this condition-test result."""
        tree = super().to_output_tree(
            include_epochs=include_epochs,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        _deep_merge(
            tree,
            {
                "data": {
                    "signal_activity": {
                        "difference": {
                            "mean": self.difference.mean.astype(np.float64),
                            "sem": self.difference.sem.astype(np.float64),
                            "ci95_low": self.difference.ci95_low.astype(np.float64),
                            "ci95_high": self.difference.ci95_high.astype(np.float64),
                        },
                    },
                },
                "stats": {
                    "condition_contrast": self.contrast.to_output_dict(
                        significance_alpha=self.significance_alpha
                    ),
                },
                "meta": {
                    "analysis_type": "condition_test",
                    "n_permutations": int(self.metadata.get("n_permutations", 0)),
                    "channel_significance_mode": str(
                        self.metadata.get("channel_significance_mode", "none")
                    ),
                    "channel_significance_duration_threshold_ms": float(
                        self.metadata.get(
                            "channel_significance_duration_threshold_ms",
                            100.0,
                        )
                    ),
                },
            },
        )
        return tree

