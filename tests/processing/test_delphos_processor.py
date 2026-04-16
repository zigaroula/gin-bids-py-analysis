from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gin_bids_py_analysis.processing.delphos.params import DelphosParams
from gin_bids_py_analysis.processing.delphos.processor import DelphosProcessing
from gin_bids_py_analysis.processing.delphos.result import DelphosProcessingResult
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)


class _FakeRaw:
    def __init__(self, data: np.ndarray, ch_names: list[str], sfreq: float) -> None:
        self._data = data
        self.ch_names = ch_names
        self.info = {"sfreq": sfreq}

    def get_data(self) -> np.ndarray:
        return self._data

    def close(self) -> None:
        return None


def test_delphos_processor_uses_threads_budget_for_pyfftw(
    mock_bids_file,
    monkeypatch,
) -> None:
    raw = _FakeRaw(
        data=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64),
        ch_names=["X01", "X02"],
        sfreq=512.0,
    )
    mock_bids_file.attach_data(raw)

    fake_pyfftw = SimpleNamespace(config=SimpleNamespace(NUM_THREADS=99))
    monkeypatch.setattr(
        "gin_bids_py_analysis.processing.delphos.processor._PYFFTW_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        "gin_bids_py_analysis.processing.delphos.processor.pyfftw",
        fake_pyfftw,
    )
    monkeypatch.setattr(
        "gin_bids_py_analysis.processing.delphos.processor.get_threads_for_worker",
        lambda: 4,
    )

    def _fake_detector(**kwargs):
        return SimpleNamespace(
            markers=[],
            n_spk=np.zeros(1, dtype=np.float64),
            n_osc=np.zeros((1, 1), dtype=np.float64),
            detection_charac=np.empty((0,), dtype=object),
            event_rates={},
        )

    monkeypatch.setattr(
        "gin_bids_py_analysis.processing.delphos.processor.delphos_detector",
        _fake_detector,
    )

    processor = DelphosProcessing(
        DelphosParams(
            detection_type=["Osc", "Spk"],
            montage_mode=MontageMode.BIPOLAR,
            bipolar_direction=BipolarDirection.NEXT_MINUS_PREVIOUS,
            bipolar_storage=BipolarStorage.NEXT,
            channels_for_montage={"01": ["X01", "X02"]},
        ),
        verbose=False,
    )

    result = processor.process_file(mock_bids_file)

    assert isinstance(result, DelphosProcessingResult)
    assert fake_pyfftw.config.NUM_THREADS == 4
    assert result.channel_names == ["X02"]
