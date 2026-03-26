"""Entry point for the trial statistics visualization application."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gin_bids_py_analysis.bids import BIDSFileGroup
    from gin_bids_py_analysis.processing.trial_stats import (
        TrialLabelResolver,
        TrialStatsParams,
    )


def launch(
    subject_groups: dict[str, "BIDSFileGroup"],
    params: "TrialStatsParams",
    resolver: "TrialLabelResolver",
) -> None:
    """Launch the trial statistics visualization window.

    Parameters
    ----------
    subject_groups:
        Mapping of ``subject_id → BIDSFileGroup`` for all subjects to visualize.
    params:
        Default ``TrialStatsParams`` to pre-populate the parameters panel.
    resolver:
        The ``TrialLabelResolver`` configured for this dataset.
    """
    try:
        import sys

        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise ImportError(
            "PySide6 is required for visualization. "
            "Install with: pip install 'gin-bids-py-analysis[viz]'"
        ) from exc

    app = QApplication.instance() or QApplication(sys.argv)

    from .window import TrialStatsWindow

    window = TrialStatsWindow(subject_groups, params, resolver)
    window.show()
    app.exec()
