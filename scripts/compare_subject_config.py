"""Subject configuration for comparison scripts.

Edit the two lines below to switch subjects.  All paths are derived from
the subject identifier automatically.

Subject format: ``GRE_2022_BRUp`` (underscores allowed).
The BIDS subject label is obtained by removing underscores: ``GRE2022BRUp``.

``BV_EVENT_SAMPLE_SHIFT_SAMPLES`` controls how compare_b1.py and compare_b2.py
align the BrainVision Hilbert derivative events to MATLAB b1/b2 epoch anchors.

This value is **-1 for all subjects and formats**.  gin2bids is configured
so that BIDS event onsets match the SPM ``event.time`` convention used by the
MATLAB a1 converters (ImaGIN for Micromed, spm_eeg_convert_Prague_data for
Prague).  Both converters store event times as 1-based sample indices, which
places them one sample later than the 0-based BrainVision derivative grid.
Subtracting one sample reproduces MATLAB's epoch anchor.

Corresponding gin2bids settings in ``config_clarissa.json``:

* Micromed : ``event_sample_offset_samples=0``
* Prague   : ``event_sample_offset_samples=1``
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Edit these two lines to switch subjects
# ---------------------------------------------------------------------------

SUBJECT: str = "GRE_3021_GUIa"
BV_EVENT_SAMPLE_SHIFT_SAMPLES: int = -1
COMPARE_A_SPM_EVENT_SAMPLE_SHIFT_SAMPLES: int = 0

# ---------------------------------------------------------------------------
# Derived identifiers (computed — do not edit)
# ---------------------------------------------------------------------------

BIDS_SUBJECT: str = SUBJECT.replace("_", "")

# ---------------------------------------------------------------------------
# Base roots (edit if the data lives somewhere else)
# ---------------------------------------------------------------------------

BIDS_ROOT: Path = Path(r"D:\Boulot\clarissa_bids")
MATLAB_ROOT: Path = Path(r"C:\GRE\dev\clarissa\seeg")

# ---------------------------------------------------------------------------
# Derived paths (computed — do not edit)
# ---------------------------------------------------------------------------

# MATLAB a3 continuous SPM file after event correction and channel renaming.
MATLAB_A3_PATH: Path = (
    MATLAB_ROOT / "a3_channels_config"
    / f"{SUBJECT}_MD_CHOICE_Partie1.mat"
)

# BIDS raw BrainVision file and event table.
BIDS_RAW_EEG_PATH: Path = (
    BIDS_ROOT / f"sub-{BIDS_SUBJECT}" / "ieeg"
    / f"sub-{BIDS_SUBJECT}_task-MDCHOICE_ieeg.eeg"
)

BIDS_RAW_EVENTS_TSV_PATH: Path = (
    BIDS_ROOT / f"sub-{BIDS_SUBJECT}" / "ieeg"
    / f"sub-{BIDS_SUBJECT}_task-MDCHOICE_events.tsv"
)

# MATLAB b1 epochs file  (used by compare_b1_vs_bids_raw.py)
MATLAB_B1_PATH: Path = (
    MATLAB_ROOT / "b1_BPF_data"
    / f"ed{SUBJECT}_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_onset.mat"
)

# MATLAB b2 cleaned-epochs file  (used by compare_matlab_vs_bids.py
#                                  and compare_b3_vs_python_regression.py)
MATLAB_B2_PATH: Path = (
    MATLAB_ROOT / "b2_BPF_apply_options"
    / f"ed{SUBJECT}_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_onset.mat"
)

# MATLAB b3 regressions folder  (used by compare_b3_vs_python_regression.py)
MATLAB_B3_ROOT: Path = (
    MATLAB_ROOT / "b3_BPF_indiv_analyses" / SUBJECT / "regressions"
)

# BrainVision Hilbert derivative  (used by compare_b1_vs_bids_raw.py
#                                   and compare_matlab_vs_bids.py)
BV_EEG_PATH: Path = (
    BIDS_ROOT / "derivatives" / "hilbert"
    / f"sub-{BIDS_SUBJECT}" / "ieeg"
    / f"sub-{BIDS_SUBJECT}_task-MDCHOICE_desc-bgasm250_ieeg.eeg"
)

# Python regression HDF5 output  (used by compare_b3_vs_python_regression.py)
PYTHON_REGRESSION_PATH: Path = (
    BIDS_ROOT / "derivatives" / "regression"
    / f"sub-{BIDS_SUBJECT}" / "ieeg"
    / f"sub-{BIDS_SUBJECT}_task-MDCHOICE_desc-onset_stats.h5"
)

# Behavior TSV  (used by compare_b2.py and compare_b3_vs_python_regression.py)
BEHAVIOR_TSV_PATH: Path = (
    BIDS_ROOT / f"sub-{BIDS_SUBJECT}" / "beh"
    / f"sub-{BIDS_SUBJECT}_task-MDCHOICE_beh.tsv"
)

# MATLAB b2 baseline-info file  (used by compare_b2.py)
MATLAB_BSL_INFO_PATH: Path = (
    MATLAB_ROOT / "b2_BPF_apply_options"
    / f"ed{SUBJECT}_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_bsl_info.mat"
)
