from __future__ import annotations

from pathlib import Path

from scripts.run_hfo_spike_detection import (
    _extract_first_bipolar_contact,
    _extract_second_bipolar_contact,
    _filter_files_with_subject_channels,
    _load_subject_channels_from_csv,
)


def test_compact_bipolar_pair_extracts_both_contacts() -> None:
    assert _extract_first_bipolar_contact("Xp02Xp01") == "Xp02"
    assert _extract_second_bipolar_contact("Xp02Xp01") == "Xp01"


def test_load_subject_channels_from_csv_keeps_both_contacts_for_compact_pairs(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "channels.csv"
    csv_path.write_text(
        "subname,channel\n"
        "GRE_2022_NASo,Xp02Xp01\n"
        "GRE_2022_NASo,Xp03Xp02\n",
        encoding="utf-8",
    )

    loaded = _load_subject_channels_from_csv(csv_path)

    assert loaded["GRE2022NASo"] == ["Xp02", "Xp01", "Xp03"]


def test_load_subject_channels_from_csv_ignores_nan_only_in_extra_columns(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "channels_with_stats.csv"
    csv_path.write_text(
        "subname,channel,beta\n"
        "GRE_2022_NASo,R03R02,nan\n",
        encoding="utf-8",
    )

    loaded = _load_subject_channels_from_csv(csv_path)

    assert loaded["GRE2022NASo"] == ["R03", "R02"]


def test_filter_files_with_subject_channels_skips_missing_subjects() -> None:
    class _File:
        def __init__(self, subject: str) -> None:
            self.subject = subject

        def get(self, key: str) -> str | None:
            if key == "subject":
                return self.subject
            return None

    kept = _filter_files_with_subject_channels(
        [_File("S01"), _File("S02"), _File("S03")],
        {"S01": ["A1"], "S02": []},
    )

    assert [file.subject for file in kept] == ["S01"]
