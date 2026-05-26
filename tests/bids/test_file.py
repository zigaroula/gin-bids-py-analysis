"""
Tests for BIDSFile entity access and lazy loading API.

These tests use the ``mock_bids_file`` fixture (no pybids indexing required).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile


def test_entity_access_via_getitem(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file["subject"] == "01"
    assert mock_bids_file["task"] == "rest"


def test_entity_access_via_get(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.get("run") == "1"
    assert mock_bids_file.get("nonexistent", "default") == "default"
    assert mock_bids_file.get("nonexistent") is None


def test_entities_returns_dict(mock_bids_file: BIDSFile) -> None:
    ents = mock_bids_file.entities
    assert isinstance(ents, dict)
    assert "subject" in ents


def test_entities_is_a_copy(mock_bids_file: BIDSFile) -> None:
    ents = mock_bids_file.entities
    ents["subject"] = "mutated"
    assert mock_bids_file["subject"] == "01"


def test_suffix_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.suffix == "ieeg"


def test_extension_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.extension == ".vhdr"


def test_datatype_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.datatype == "ieeg"


def test_missing_entity_raises_key_error(mock_bids_file: BIDSFile) -> None:
    with pytest.raises(KeyError, match="nonexistent_entity"):
        _ = mock_bids_file["nonexistent_entity"]


def test_repr(mock_bids_file: BIDSFile) -> None:
    assert "BIDSFile" in repr(mock_bids_file)


def test_from_path_alternate_constructor() -> None:
    file = BIDSFile.from_path(
        Path("sub-01/ses-02/ieeg/sub-01_ses-02_task-rest_run-1_ieeg.vhdr"),
    )

    assert file.path == Path("sub-01/ses-02/ieeg/sub-01_ses-02_task-rest_run-1_ieeg.vhdr")
    assert file["sub"] == "01"
    assert file["subject"] == "01"
    assert file["ses"] == "02"
    assert file["session"] == "02"
    assert file["task"] == "rest"
    assert file.extension == ".vhdr"
    assert file.datatype == "ieeg"
    assert file.suffix == "ieeg"


# ---------------------------------------------------------------------------
# Loaded-data encapsulation
# ---------------------------------------------------------------------------


class TestIsLoadedAndData:
    def test_is_loaded_false_by_default(self, mock_bids_file: BIDSFile) -> None:
        assert mock_bids_file.is_loaded is False
        assert mock_bids_file.data is None

    def test_attach_data_sets_is_loaded(self, mock_bids_file: BIDSFile) -> None:
        fake_raw = MagicMock()
        mock_bids_file.attach_data(fake_raw)
        assert mock_bids_file.is_loaded is True
        assert mock_bids_file.data is fake_raw

    def test_attach_data_is_chainable(self, mock_bids_file: BIDSFile) -> None:
        fake_raw = MagicMock()
        result = mock_bids_file.attach_data(fake_raw)
        assert result is mock_bids_file

    def test_set_loader_is_chainable(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        result = mock_bids_file.set_loader(fake_loader)
        assert result is mock_bids_file


class TestEnsureLoadedExternally:
    """ensure_loaded() on a file with pre-attached data."""

    def test_yields_attached_data(self, mock_bids_file: BIDSFile) -> None:
        fake_raw = MagicMock()
        mock_bids_file.attach_data(fake_raw)
        with mock_bids_file.ensure_loaded() as data:
            assert data is fake_raw

    def test_data_persists_after_context_exits(self, mock_bids_file: BIDSFile) -> None:
        fake_raw = MagicMock()
        mock_bids_file.attach_data(fake_raw)
        with mock_bids_file.ensure_loaded():
            pass
        assert mock_bids_file.is_loaded is True
        assert mock_bids_file.data is fake_raw


class TestEnsureLoadedOnDemand:
    """ensure_loaded() on an unloaded file (mock loader injected via set_loader)."""

    def test_calls_loader_and_yields_data(self, mock_bids_file: BIDSFile) -> None:
        fake_data = MagicMock()
        fake_loader = MagicMock(return_value=fake_data)
        mock_bids_file.set_loader(fake_loader)

        with mock_bids_file.ensure_loaded() as data:
            assert data is fake_data
            fake_loader.assert_called_once_with(mock_bids_file.path)

    def test_data_is_cleared_after_context_exits(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader)

        with mock_bids_file.ensure_loaded():
            assert mock_bids_file.is_loaded is True

        assert mock_bids_file.is_loaded is False
        assert mock_bids_file.data is None

    def test_set_loader_defaults_forwarded(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader, verbose=False)

        with mock_bids_file.ensure_loaded():
            pass

        fake_loader.assert_called_once_with(mock_bids_file.path, verbose=False)

    def test_call_time_kwargs_override_defaults(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader, preload=True)

        with mock_bids_file.ensure_loaded(preload=False):
            pass

        fake_loader.assert_called_once_with(mock_bids_file.path, preload=False)

    def test_data_cleared_even_if_context_body_raises(
        self, mock_bids_file: BIDSFile
    ) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader)

        with pytest.raises(RuntimeError):
            with mock_bids_file.ensure_loaded():
                raise RuntimeError("oops")

        assert mock_bids_file.is_loaded is False


class TestPreload:
    """preload() loads data permanently so ensure_loaded() returns it without re-reading."""

    def test_preload_attaches_data(self, mock_bids_file: BIDSFile) -> None:
        fake_data = MagicMock()
        mock_bids_file.set_loader(MagicMock(return_value=fake_data))
        mock_bids_file.preload()
        assert mock_bids_file.is_loaded is True
        assert mock_bids_file.data is fake_data

    def test_preload_is_chainable(self, mock_bids_file: BIDSFile) -> None:
        mock_bids_file.set_loader(MagicMock(return_value=MagicMock()))
        result = mock_bids_file.preload()
        assert result is mock_bids_file

    def test_ensure_loaded_reuses_preloaded_data(self, mock_bids_file: BIDSFile) -> None:
        fake_data = MagicMock()
        fake_loader = MagicMock(return_value=fake_data)
        mock_bids_file.set_loader(fake_loader)
        mock_bids_file.preload()

        with mock_bids_file.ensure_loaded() as data:
            assert data is fake_data

        # Loader only called once (during preload, not during ensure_loaded)
        fake_loader.assert_called_once()

    def test_data_persists_after_ensure_loaded_exits(self, mock_bids_file: BIDSFile) -> None:
        mock_bids_file.set_loader(MagicMock(return_value=MagicMock()))
        mock_bids_file.preload()

        with mock_bids_file.ensure_loaded():
            pass

        # Data must still be attached — not cleared by ensure_loaded's finally
        assert mock_bids_file.is_loaded is True

    def test_preload_is_noop_when_already_loaded(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader)
        mock_bids_file.preload()
        mock_bids_file.preload()  # second call
        fake_loader.assert_called_once()

    def test_preload_forwards_kwargs(self, mock_bids_file: BIDSFile) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader, verbose=False)
        mock_bids_file.preload(extra_kwarg=True)
        fake_loader.assert_called_once_with(
            mock_bids_file.path, verbose=False, extra_kwarg=True
        )

    def test_multiple_ensure_loaded_calls_do_not_reload(
        self, mock_bids_file: BIDSFile
    ) -> None:
        fake_loader = MagicMock(return_value=MagicMock())
        mock_bids_file.set_loader(fake_loader)
        mock_bids_file.preload()

        for _ in range(5):
            with mock_bids_file.ensure_loaded():
                pass

        fake_loader.assert_called_once()


class TestAutoDetection:
    """Auto-detection of loader from file extension."""

    def _file_with_ext(self, ext: str) -> BIDSFile:
        return BIDSFile.from_path(
            Path(f"sub-01/ses-01/ieeg/sub-01_ses-01_task-rest_ieeg{ext}")
        )

    def test_vhdr_uses_load_ieeg(self) -> None:
        file = self._file_with_ext(".vhdr")
        fake_raw = MagicMock()
        with patch(
            "gin_bids_py_analysis.data.loader.mne.io.read_raw", return_value=fake_raw
        ):
            with file.ensure_loaded() as data:
                assert data is fake_raw

    def test_tsv_uses_load_table(self, tmp_path: Path) -> None:
        tsv = tmp_path / "sub-01_electrodes.tsv"
        tsv.write_text("name\tregion\nA1\tFrontal\n", encoding="utf-8")
        file = BIDSFile.from_path(tsv)
        with file.ensure_loaded() as rows:
            assert rows == [{"name": "A1", "region": "Frontal"}]

    def test_json_uses_load_json(self, tmp_path: Path) -> None:
        jf = tmp_path / "sub-01_ieeg.json"
        jf.write_text('{"SamplingFrequency": 1000}', encoding="utf-8")
        file = BIDSFile.from_path(jf)
        with file.ensure_loaded() as meta:
            assert meta == {"SamplingFrequency": 1000}

    def test_unknown_extension_raises_value_error(self) -> None:
        file = BIDSFile.from_path(Path("sub-01/ses-01/ieeg/sub-01_ieeg.xyz"))
        with pytest.raises(ValueError, match=r"No loader registered"):
            with file.ensure_loaded():
                pass


class TestPickleSafety:
    def test_data_dropped_on_pickle(self, mock_bids_file: BIDSFile) -> None:
        mock_bids_file.attach_data({"some": "data"})
        assert mock_bids_file.is_loaded is True

        roundtripped = pickle.loads(pickle.dumps(mock_bids_file))

        assert roundtripped.is_loaded is False
        assert roundtripped.data is None

    def test_externally_loaded_reset_on_pickle(self, mock_bids_file: BIDSFile) -> None:
        mock_bids_file.attach_data(MagicMock())
        assert mock_bids_file._externally_loaded is True

        roundtripped = pickle.loads(pickle.dumps(mock_bids_file))

        assert roundtripped._externally_loaded is False

    def test_loader_and_kwargs_survive_pickle(self, mock_bids_file: BIDSFile) -> None:
        from gin_bids_py_analysis.data.loader import load_table  # noqa: PLC0415

        mock_bids_file.set_loader(load_table, verbose=False)
        roundtripped = pickle.loads(pickle.dumps(mock_bids_file))

        assert roundtripped._loader is load_table
        assert roundtripped._load_kwargs == {"verbose": False}

    def test_path_and_entities_survive_pickle(self, mock_bids_file: BIDSFile) -> None:
        roundtripped = pickle.loads(pickle.dumps(mock_bids_file))

        assert roundtripped.path == mock_bids_file.path
        assert roundtripped.entities == mock_bids_file.entities



