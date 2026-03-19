"""Reusable helpers for BIDS TSV/CSV table loading and cell coercion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Sequence

from gin_bids_py_analysis.bids.file import BIDSFile

_TABLE_DELIMITERS: dict[str, str] = {
    ".tsv": "\t",
    ".csv": ",",
}


@dataclass(frozen=True)
class LoadedTableRow:
    values: dict[str, str]
    file: BIDSFile
    row_index: int


def is_supported_table(file: BIDSFile) -> bool:
    return file.extension in _TABLE_DELIMITERS


def read_table_rows(file: BIDSFile) -> list[dict[str, str]]:
    delimiter = _TABLE_DELIMITERS.get(file.extension)
    if delimiter is None:
        raise ValueError(
            f"Unsupported table extension for {file.path.name!r}: {file.extension!r}"
        )

    with open(file.path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]


def load_table_rows(files: Sequence[BIDSFile]) -> list[LoadedTableRow]:
    rows: list[LoadedTableRow] = []
    for file in files:
        if not is_supported_table(file):
            continue
        for idx, values in enumerate(read_table_rows(file)):
            rows.append(LoadedTableRow(values=values, file=file, row_index=idx))
    return rows


def row_value(row: LoadedTableRow | None, column: str | None) -> str | None:
    if row is None or column is None:
        return None
    return row.values.get(column)


def row_float_value(row: LoadedTableRow | None, column: str | None) -> float | None:
    raw_value = row_value(row, column)
    if raw_value in (None, ""):
        return None
    return float(raw_value)


def row_int_value(row: LoadedTableRow | None, column: str | None) -> int | None:
    raw_value = row_value(row, column)
    if raw_value in (None, ""):
        return None
    return int(raw_value)

def row_code_value(row: LoadedTableRow | None, column: str | None) -> str | None:
    raw_value = row_value(row, column)
    if raw_value is None or raw_value == "":
        return None
    return str(int(raw_value)) if raw_value.isdigit() else raw_value

def select_column(columns: Sequence[str], preferred: Sequence[str]) -> str | None:
    by_norm = {column.casefold(): column for column in columns}
    for wanted in preferred:
        match = by_norm.get(wanted.casefold())
        if match is not None:
            return match
    return None

