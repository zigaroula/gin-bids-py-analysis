from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.processing.utils.tables import (
    LoadedTableRow,
    load_table_rows,
    row_float_value,
    row_int_value,
    row_value,
    select_column,
)


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def test_load_table_rows_trims_headers_and_values(tmp_path: Path) -> None:
    table_path = tmp_path / "sub-01_events.tsv"
    table_path.write_text(" trial_id \t onset \n trial-1 \t 1.0 \n", encoding="utf-8")
    table_file = _make_bids_file(
        table_path,
        {"extension": ".tsv", "suffix": "events"},
    )

    rows = load_table_rows([table_file])
    assert len(rows) == 1
    assert rows[0].values == {"trial_id": "trial-1", "onset": "1.0"}


def test_load_table_rows_skips_unsupported_extensions(tmp_path: Path) -> None:
    table_path = tmp_path / "events.tsv"
    table_path.write_text("trial_id\ntrial-1\ntrial-2\n", encoding="utf-8")
    ignored_path = tmp_path / "notes.txt"
    ignored_path.write_text("ignored", encoding="utf-8")

    table_file = _make_bids_file(table_path, {"extension": ".tsv", "suffix": "events"})
    ignored_file = _make_bids_file(ignored_path, {"extension": ".txt", "suffix": "events"})

    rows = load_table_rows([table_file, ignored_file])

    assert [row.row_index for row in rows] == [0, 1]
    assert [row.values["trial_id"] for row in rows] == ["trial-1", "trial-2"]
    assert all(row.file.path == table_path for row in rows)


def test_row_value_and_numeric_coercions() -> None:
    fake_file = _make_bids_file(
        Path("/tmp/sub-01_events.tsv"),
        {"extension": ".tsv", "suffix": "events"},
    )
    row = LoadedTableRow(
        values={"int_col": "3", "float_col": "2.5", "empty": ""},
        file=fake_file,
        row_index=0,
    )

    assert row_value(row, "int_col") == "3"
    assert row_int_value(row, "int_col") == 3
    assert row_float_value(row, "float_col") == 2.5
    assert row_int_value(row, "empty") is None
    assert row_float_value(None, "float_col") is None
    assert row_value(None, "int_col") is None


def test_select_column_is_case_insensitive() -> None:
    columns = ["Name", "MarsAtlas", "Other"]

    assert select_column(columns, preferred=["name"]) == "Name"
    assert select_column(columns, preferred=["marsatlas"]) == "MarsAtlas"
    assert select_column(columns, preferred=["missing"]) is None


