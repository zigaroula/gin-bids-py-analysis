from __future__ import annotations

import numpy as np

try:
    import pyfftw

    _PYFFTW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYFFTW_AVAILABLE = False

from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.base import BaseProcessing
from bidsforge.processing.utils.channels import build_montage, select_channels_for_montage
from bidsforge.processing.utils.epoching import (
    extract_anchor_events_with_mne,
    extract_epochs_with_mne,
)
from bidsforge.processing.utils.input_events import resolve_input_events
from bidsforge.processing.utils.multithreading import get_threads_for_worker
from bidsforge.processing.utils.trial_resolver import ResolvedTrial

from .dsp import apply_baseline_correction, compute_baseline_db, compute_power_db
from .params import TimeFrequencyParams
from .result import TimeFrequencyProcessingResult


class TimeFrequencyProcessing(BaseProcessing):
    """Compute trial-level multitaper time-frequency power."""

    def __init__(self, params: TimeFrequencyParams, verbose: bool = True) -> None:
        self.params = params
        self.verbose = verbose

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TimeFrequencyProcessingResult:
        if _PYFFTW_AVAILABLE:
            pyfftw.config.NUM_THREADS = get_threads_for_worker()

        with group.primary.ensure_loaded() as raw:
            fs = float(raw.info["sfreq"])
            data = raw.get_data().astype(np.float32)
            channel_names = list(raw.ch_names)
            resolved_events = resolve_input_events(group, raw, self.params.events_source)
            exact_event_records = (
                resolved_events.events
                if resolved_events.onset_precision == "exact_time"
                else None
            )
            anchor_events, anchor_samples = extract_anchor_events_with_mne(
                raw,
                anchor_codes={str(code) for code in self.params.anchor_event_codes},
                experiment_start_event_code=self.params.experiment_start_event_code,
                experiment_end_event_code=self.params.experiment_end_event_code,
                raw_annotations=exact_event_records,
                event_sample_shift_samples=self.params.event_sample_shift_samples,
            )

        data, channel_names = select_channels_for_montage(
            data,
            channel_names,
            self.params.channels_for_montage,
            self.params.channels_to_exclude_for_montage,
        )
        data, channel_names = build_montage(
            data=data,
            channel_names=channel_names,
            mode=self.params.montage_mode,
            direction=self.params.bipolar_direction,
            storage=self.params.bipolar_storage,
        )

        trials = [
            ResolvedTrial(
                source_file=group.primary,
                anchor_event_index=idx,
                anchor_event_code=str(event.code),
                anchor_onset_s=float(event.onset_s),
                anchor_duration_s=float(event.duration_s),
                label="anchor",
                trial_id=str(idx),
                keep=True,
            )
            for idx, event in enumerate(anchor_events)
        ]
        raw_for_epochs = _array_to_raw(data, channel_names, fs)
        extraction = extract_epochs_with_mne(
            raw_for_epochs,
            anchor_samples=anchor_samples,
            trials=trials,
            tmin_s=self.params.tmin_s,
            tmax_s=self.params.tmax_s,
            drop_partial_epochs=self.params.drop_partial_epochs,
        )

        power_raw, grid = compute_power_db(
            extraction.epochs,
            extraction.time_axis_s,
            fs,
            self.params,
            verbose=self.verbose,
            desc=f"TFR {group.primary.path.name}",
            progress_tracking_position=progress_tracking_position,
        )
        baseline = compute_baseline_db(
            power_raw,
            grid.time_s,
            self.params.baseline_window_s,
        )
        power = apply_baseline_correction(power_raw, baseline) if self.params.apply_baseline else power_raw

        trial_ids = [
            str(getattr(trial, "trial_id", idx))
            for idx, trial in enumerate(extraction.kept_trials)
        ]
        fieldtrip_pad_s = (
            float(extraction.epochs.shape[-1]) / fs
            if self.params.fieldtrip_pad_s is None
            else float(self.params.fieldtrip_pad_s)
        )
        return TimeFrequencyProcessingResult(
            source_group=group,
            power_db=power,
            baseline_db=baseline,
            trial_ids=trial_ids,
            channel_names=channel_names,
            frequency_hz=grid.frequency_hz,
            time_s=grid.time_s,
            original_fs=fs,
            events=resolved_events.events,
            metadata={
                "apply_baseline": bool(self.params.apply_baseline),
                "baseline_window_s": list(self.params.baseline_window_s),
                "time_decimation": int(self.params.time_decimation),
                "time_frequency_method": self.params.method.value,
                "fieldtrip_polyorder": int(self.params.fieldtrip_polyorder),
                "fieldtrip_pad_s": fieldtrip_pad_s,
                "fieldtrip_padtype": self.params.fieldtrip_padtype,
                "fieldtrip_scaling": self.params.method.value == "fieldtrip",
                "montage_mode": self.params.montage_mode.value,
                "events_source_requested": self.params.events_source,
                "events_source_resolved": resolved_events.source_resolved,
                "events_onset_precision": resolved_events.onset_precision,
                "event_sample_shift_samples": int(self.params.event_sample_shift_samples),
                "pyfftw_available": bool(_PYFFTW_AVAILABLE),
            },
        )


def _array_to_raw(data: np.ndarray, channel_names: list[str], fs: float):
    import mne

    info = mne.create_info(channel_names, sfreq=fs, ch_types=["seeg"] * len(channel_names))
    return mne.io.RawArray(data, info, verbose=False)
