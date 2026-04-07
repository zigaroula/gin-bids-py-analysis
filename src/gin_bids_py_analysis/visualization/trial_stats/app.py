"""Entry point for the trial statistics visualization application."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gin_bids_py_analysis.bids import BIDSFileGroup
    from gin_bids_py_analysis.processing.trial_slope_stats import (
        TrialSlopeStatsParams,
    )
    from gin_bids_py_analysis.processing.trial_stats import (
        TrialResolver,
        TrialStatsParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.result import (
        TrialStatsGroupProcessingResult,
    )


def launch(
    subject_groups: dict[str, "BIDSFileGroup"],
    params: "TrialStatsParams",
    resolver: "TrialResolver",
    group_params: "TrialStatsGroupParams | None" = None,
    bids_root: Path | None = None,
) -> None:
    """Launch the trial statistics visualization window.

    Parameters
    ----------
    subject_groups:
        Mapping of ``subject_id → BIDSFileGroup`` for all subjects to visualize.
    params:
        Default ``TrialStatsParams`` to pre-populate the parameters panel.
    resolver:
        The ``TrialResolver`` configured for this dataset.
    group_params:
        Optional ``TrialStatsGroupParams``.  When provided the Group tab is
        enabled after all subjects have been computed.
    bids_root:
        Path to the BIDS dataset root.  When provided the *Save results* buttons
        in both the Subject and Group tabs become functional after a successful
        compute, allowing results to be written to the BIDS derivatives folder.
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

    window = TrialStatsWindow(subject_groups, params, resolver, group_params=group_params, bids_root=bids_root)
    window.show()
    app.exec()


def launch_slope(
    subject_groups: dict[str, "BIDSFileGroup"],
    params: "TrialSlopeStatsParams",
    resolver: "TrialResolver",
    bids_root: Path | None = None,
) -> None:
    """Launch the trial statistics visualization window in slope mode."""
    try:
        import sys

        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise ImportError(
            "PySide6 is required for visualization. "
            "Install with: pip install 'gin-bids-py-analysis[viz]'"
        ) from exc

    from gin_bids_py_analysis.processing.trial_stats import TrialStatsParams

    app = QApplication.instance() or QApplication(sys.argv)

    from .window import TrialStatsWindow

    fallback_ttest_params = TrialStatsParams(
        anchor_event_codes=list(params.anchor_event_codes),
        tmin_s=params.tmin_s,
        tmax_s=params.tmax_s,
        condition_a=params.condition_a,
        condition_b=params.condition_b,
        min_trials_per_condition=max(2, params.min_trials_per_condition),
        drop_partial_epochs=params.drop_partial_epochs,
        p_value_correction_method=params.p_value_correction_method,
        significance_alpha=params.significance_alpha,
        atlas_name=params.atlas_name,
        atlas_regions=list(params.atlas_regions),
        window_ms=params.window_ms,
        n_bins=params.n_bins,
    )
    window = TrialStatsWindow(
        subject_groups,
        fallback_ttest_params,
        resolver,
        group_params=None,
        bids_root=bids_root,
        default_slope_params=params,
        default_mode="slope",
    )
    window.show()
    app.exec()


def launch_precomputed(
    subject_stats_files: dict[str, Path],
    group_stats_file: Path | None = None,
    group_params: "TrialStatsGroupParams | None" = None,
) -> None:
    """Launch the trial statistics viewer loading pre-computed result files.

    Use this entry point when the per-subject statistics have already been
    written to disk by ``TrialStatsProcessingWriter`` (for example after a slow
    permutation run) and you want to visualize the saved results without
    re-running the pipeline.

    Parameters
    ----------
    subject_stats_files:
        Mapping of ``subject_id → Path`` to the per-subject stats file
        (``.h5``/``.hdf5`` or ``.mat``).
    group_stats_file:
        Optional path to a pre-computed group stats file written by
        ``TrialStatsGroupProcessingWriter``.  When provided the group result is
        loaded from disk and displayed immediately in the Group tab once all
        subject files are ready.  The ``GroupParamsPanel`` is still available
        so the group analysis can be re-run with different parameters.
    group_params:
        Optional ``TrialStatsGroupParams`` used to pre-populate the
        ``GroupParamsPanel``.  When *group_stats_file* is ``None`` and
        *group_params* is set, the user can click *Compute group stats* to run
        the group analysis from the loaded subject results.
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

    from .window_precomputed import TrialStatsPrecomputedWindow

    window = TrialStatsPrecomputedWindow(
        subject_stats_files,
        group_file=group_stats_file,
        group_params=group_params,
    )
    window.show()
    app.exec()


def launch_group_precomputed(
    group_stats_file: Path,
    group_params: "TrialStatsGroupParams | None" = None,
) -> None:
    """Launch a group-only viewer loading a pre-computed group stats file.

    Use this entry point when you only have a group stats file written by
    ``TrialStatsGroupProcessingWriter`` and do not need to visualize individual
    subjects.  The window shows the ``GroupPlotPanel`` immediately after the
    file is loaded.

    Parameters
    ----------
    group_stats_file:
        Path to the group stats file (``.h5``/``.hdf5``).
    group_params:
        Optional ``TrialStatsGroupParams`` used to pre-populate the
        ``GroupParamsPanel`` display.  No computation is performed — the
        params are shown for reference only.
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

    from .window_group_precomputed import TrialStatsGroupPrecomputedWindow

    window = TrialStatsGroupPrecomputedWindow(
        group_stats_file,
        group_params=group_params,
    )
    window.show()
    app.exec()
