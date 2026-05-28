"""
Convert precomputed regression result files to another writer format.

Edit the settings below, then run from the repository root:

    python scripts/export_regression_results_format.py

The default use case is:
- find all subject-level and group-level regression ``*_stats.h5`` files
- load them with the matching result loader
- re-export them with the matching writer as MATLAB ``.mat`` files
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py

from bidsforge.processing.trial_stats.regression.result_loader import (
    load_regression_result,
)
from bidsforge.processing.trial_stats.regression.writer import (
    RegressionProcessingWriter,
)
from bidsforge.processing.trial_stats.regression.params import (
    RegressionWriterParams,
)
from bidsforge.processing.trial_stats_group.regression.result_loader import (
    load_regression_group_result,
)
from bidsforge.processing.trial_stats_group.regression.writer import (
    RegressionGroupProcessingWriter,
)
from bidsforge.processing.trial_stats_group.regression.params import (
    RegressionGroupWriterParams,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from trial_slope_shared import BIDS_ROOT as DEFAULT_BIDS_ROOT
except Exception:
    DEFAULT_BIDS_ROOT = Path(".")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_BIDS_ROOT = Path(DEFAULT_BIDS_ROOT)
OUTPUT_BIDS_ROOT = Path(DEFAULT_BIDS_ROOT)

SEARCH_ROOTS = (
    INPUT_BIDS_ROOT / "derivatives",
)
INPUT_GLOB = "**/sub-group_*_stats.h5"

INCLUDE_SUBJECT_REGRESSION = True
INCLUDE_GROUP_REGRESSION = True

OUTPUT_FORMAT = "matlab"
INCLUDE_EPOCHS = False

SKIP_EXISTING = True
DRY_RUN = False
STOP_ON_ERROR = False
PRESERVE_INPUT_DESC = True


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionTarget:
    path: Path
    kind: str


def main() -> list[Path]:
    targets = _find_regression_hdf5_files()
    print(f"Found {len(targets)} regression HDF5 file(s).")
    if DRY_RUN:
        for target in targets:
            print(f"[dry-run] {target.kind:7s} {target.path}")
        return []

    out_paths: list[Path] = []
    for target in targets:
        try:
            out_path = _convert_one(target)
        except Exception as exc:
            message = f"Failed {target.kind} {target.path}: {exc}"
            if STOP_ON_ERROR:
                raise RuntimeError(message) from exc
            print(message)
            continue
        if out_path is not None:
            out_paths.append(out_path)
            print(f"Wrote {out_path}")
    return out_paths


def _find_regression_hdf5_files() -> list[ConversionTarget]:
    targets: list[ConversionTarget] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        for path in sorted(Path(root).glob(INPUT_GLOB)):
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)

            kind = _classify_regression_file(path)
            if kind == "subject" and INCLUDE_SUBJECT_REGRESSION:
                targets.append(ConversionTarget(path=path, kind=kind))
            elif kind == "group" and INCLUDE_GROUP_REGRESSION:
                targets.append(ConversionTarget(path=path, kind=kind))
    return targets


def _classify_regression_file(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as fh:
            if "regression" in fh and _h5_string(fh, "meta/analysis_type") == "slope_regression":
                return "subject"
            if "stats/regression" in fh and _h5_string(fh, "meta/analysis_type") == "regression_group":
                return "group"
    except OSError:
        return None
    return None


def _convert_one(target: ConversionTarget) -> Path | None:
    if target.kind == "subject":
        result = load_regression_result(target.path)
        writer = RegressionProcessingWriter(
            RegressionWriterParams(
                bids_root=OUTPUT_BIDS_ROOT,
                output_format=OUTPUT_FORMAT,
                output_description=_output_description(
                    target.path,
                    fallback="regression",
                ),
                include_epochs=INCLUDE_EPOCHS,
            )
        )
    elif target.kind == "group":
        result = load_regression_group_result(target.path)
        writer = RegressionGroupProcessingWriter(
            RegressionGroupWriterParams(
                bids_root=OUTPUT_BIDS_ROOT,
                output_format=OUTPUT_FORMAT,
                output_description=_output_description(
                    target.path,
                    fallback="regressiongroup",
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported conversion target kind: {target.kind!r}")

    expected_path = writer.get_output_path(result.source_group)
    if SKIP_EXISTING and expected_path.exists():
        print(f"Skipping existing {expected_path}")
        return expected_path
    return writer.write(result)


def _output_description(path: Path, *, fallback: str) -> str:
    if not PRESERVE_INPUT_DESC:
        return fallback
    for part in path.stem.split("_"):
        if part.startswith("desc-"):
            desc = part.removeprefix("desc-").strip()
            if desc:
                return desc
    return fallback


def _h5_string(fh: h5py.File, key: str) -> str:
    if key not in fh:
        return ""
    value = fh[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


if __name__ == "__main__":
    main()

