"""
Delphos HFO/spike detection processor.

Handles orchestration: loading iEEG data, applying montage, delegating to
the pure detection algorithm, and assembling results.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

try:
    import pyfftw
    _PYFFTW_AVAILABLE = True
except ImportError:
    _PYFFTW_AVAILABLE = False

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import build_montage

from .params import DelphosParams
from .result import DelphosProcessingResult
from .delphos_detector import delphos_detector
from ..utils.channels import select_channels_for_montage
from ..utils.multithreading import get_threads_for_worker


class DelphosProcessing(BaseProcessing):
    """
    Delphos HFO/spike detection processor.

    Instantiate with a DelphosParams object; then call run() (process + write)
    or execute() (process only) with a list of BIDS files or file groups.

    The processor:
    1. Loads iEEG data via MNE (format auto-detected)
    2. Applies optional channel selection
    3. Applies montage (mono or bipolar)
    4. Delegates to the pure detection algorithm (delphos_detector)
    5. Assembles a DelphosProcessingResult with detected events

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.delphos import (
            DelphosParams,
            DelphosProcessing,
            DelphosProcessingWriter,
            DelphosWriterParams,
        )

        params = DelphosParams(
            alpha=0.005,
            detection_type=["Osc", "Spk"],
            freq_band=[[80, 250], [250, 500]],
        )
        processor = DelphosProcessing(params)
        writer = DelphosProcessingWriter(
            DelphosWriterParams(bids_root=Path("/data"))
        )
        out_paths = processor.run(ieeg_files, writer)
    """

    def __init__(self, params: DelphosParams, verbose: bool = True) -> None:
        """
        Initialize the Delphos processor.

        Args:
            params: Algorithm parameters
            verbose: Whether to print progress messages
        """
        self.params = params
        self.verbose = verbose

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> DelphosProcessingResult:
        """
        Run the Delphos detection pipeline on one file group.

        Reads the iEEG file via MNE (format auto-detected from extension),
        extracts the sampling frequency and channel data, applies montage,
        runs the detection algorithm, and returns a DelphosProcessingResult.

        Args:
            group: BIDS file group with primary iEEG file
            progress_tracking_position: Position for progress tracking (joblib)

        Returns:
            DelphosProcessingResult containing detected events and metadata
        """
        # Load iEEG data
        with group.primary.ensure_loaded() as raw:
            original_fs = raw.info["sfreq"]
            channel_names = list(raw.ch_names)
            data = raw.get_data()  # (n_channels, n_samples)

        if _PYFFTW_AVAILABLE:
            pyfftw.config.NUM_THREADS = get_threads_for_worker()

        # Optional channel include/exclude selection before montage
        data, channel_names = select_channels_for_montage(
            data,
            channel_names,
            self.params.channels_for_montage,
            self.params.channels_to_exclude_for_montage,
        )

        # Apply montage (mono or bipolar)
        data, channel_names = build_montage(
            data=data,
            channel_names=channel_names,
            mode=self.params.montage_mode,
            direction=self.params.bipolar_direction,
            storage=self.params.bipolar_storage,
        )

        # Call the pure detection algorithm
        freq_band = np.array(self.params.freq_band)
        n_channels = len(channel_names)

        pbar = tqdm(
            total=n_channels,
            desc=group.primary.path.name,
            unit="ch",
            disable=not self.verbose,
            leave=True,
            position=progress_tracking_position,
        )

        def _progress_callback(pct: float, step_desc: str, current_channel: int) -> None:
            if current_channel < n_channels:
                pbar.set_postfix_str(f"{channel_names[current_channel]}: {step_desc}")
            else:
                pbar.set_postfix_str(f"{step_desc}")
            pbar.update(current_channel - pbar.n)

        detection_result = delphos_detector(
            signal=data,
            labels=channel_names,
            alpha=self.params.alpha,
            fs=original_fs,
            detection_type=self.params.detection_type,
            freq_band=freq_band,
            thr_type=self.params.thr_type,
            param_thr=np.array(self.params.param_thr) if self.params.param_thr else None,
            quantile_method=self.params.quantile_method,
            nb_voices=self.params.nb_voices,
            vanishing_moment=self.params.vanishing_moment,
            progress_callback=_progress_callback,
        )
        pbar.close()
        
        # Extract results from detection
        markers = detection_result.markers
        n_spk = detection_result.n_spk
        n_osc = detection_result.n_osc
        detection_charac = detection_result.detection_charac
        event_rates = detection_result.event_rates
        algorithm_config = self.params.model_dump()

        # Build result
        result = DelphosProcessingResult(
            source_group=group,
            metadata={
                "original_fs": original_fs,
                "n_channels": n_channels,
                "n_samples": data.shape[1],
                "detection_params": algorithm_config,
            },
            markers=markers,
            freq_band=freq_band,
            n_spk=n_spk,
            n_osc=n_osc,
            channel_names=channel_names,
            detection_charac=detection_charac,
            event_rates=event_rates,
            algorithm_config=algorithm_config,
            original_fs=original_fs,
        )

        return result
