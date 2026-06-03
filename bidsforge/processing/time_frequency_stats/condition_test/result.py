from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bidsforge.processing.utils.serialization import OutputTree, compressed

from ..result import BaseTimeFrequencyStatsResult


@dataclass
class TFDifferenceEstimate:
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TFConditionContrast:
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    permuted_t_values: np.ndarray | None = None

    def to_output_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_values": np.asarray(self.p_values, dtype=np.float64),
            "p_values_uncorrected": np.asarray(self.p_values_uncorrected, dtype=np.float64),
            "significant_mask": np.asarray(self.significant_mask, dtype=bool),
        }
        if self.permuted_t_values is not None:
            out["permuted_t_values"] = compressed(
                np.asarray(self.permuted_t_values, dtype=np.float32)
            )
        return out


@dataclass
class TFGrandAverage:
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    std: np.ndarray = field(default_factory=lambda: np.array([]))
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64),
            "std": np.asarray(self.std, dtype=np.float64),
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_values": np.asarray(self.p_values, dtype=np.float64),
        }


@dataclass
class TimeFrequencyConditionTestResult(BaseTimeFrequencyStatsResult):
    """Structured outputs for TF condition-test subject statistics."""

    difference: TFDifferenceEstimate = field(default_factory=TFDifferenceEstimate)
    contrast: TFConditionContrast = field(default_factory=TFConditionContrast)
    grand_average: TFGrandAverage | None = None

    def to_output_tree(
        self,
        *,
        include_epochs: bool = False,
        pipeline_name: str = "time_frequency_condition_test",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        tree = self._base_output_tree(
            include_epochs=include_epochs,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        tree["data"]["signal_activity"]["difference"] = {  # type: ignore[index]
            "mean": np.asarray(self.difference.mean, dtype=np.float64),
            "sem": np.asarray(self.difference.sem, dtype=np.float64),
        }
        tree["stats"] = {"condition_contrast": self.contrast.to_output_dict()}
        if self.grand_average is not None:
            tree["stats"]["grand_average"] = self.grand_average.to_output_dict()  # type: ignore[index]
        tree["meta"]["analysis_type"] = "time_frequency_condition_test"  # type: ignore[index]
        tree["meta"]["equal_var"] = bool(self.metadata.get("equal_var", False))  # type: ignore[index]
        tree["meta"]["compute_grand_average"] = bool(  # type: ignore[index]
            self.metadata.get("compute_grand_average", False)
        )
        return tree
