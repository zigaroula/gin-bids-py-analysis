from __future__ import annotations

import shutil
import uuid
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup


REPO_ROOT = Path(__file__).resolve().parents[2]


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_case_dir(case_name: str) -> Path:
    root = REPO_ROOT / "tests" / "_script_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{case_name}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _load_script_module() -> object:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_trial_slope_stats_group.py"
    spec = spec_from_file_location("run_trial_slope_stats_group", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_trial_slope_stats_group_script_main_smoke(monkeypatch) -> None:
    case_dir = _make_case_dir("run_script_slope_group_smoke")
    try:
        module = _load_script_module()

        stats_file = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-slopestat_stats.h5",
            {
                "subject": "01",
                "task": "decid",
                "desc": "slopestat",
                "suffix": "stats",
                "extension": ".h5",
            },
        )
        group = BIDSFileGroup(primary=stats_file)
        expected_out = (
            case_dir
            / "derivatives"
            / "trial_slope_stats_group"
            / "sub-group"
            / "ieeg"
            / "sub-group_task-decid_desc-trialslopestatsgroup_stats.h5"
        )

        class _FakeDataset:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def get_files(self, **_filters):
                return [stats_file]

        class _FakeProcessor:
            def __init__(self, _params) -> None:
                pass

            def run(self, groups, writer, n_jobs=1):
                del writer, n_jobs
                assert len(groups) == 1
                return [expected_out]

        class _FakeWriter:
            def __init__(self, _params) -> None:
                pass

        monkeypatch.setattr(module, "BIDSDataset", _FakeDataset)
        monkeypatch.setattr(module, "TrialSlopeStatsGroupProcessing", _FakeProcessor)
        monkeypatch.setattr(module, "TrialSlopeStatsGroupProcessingWriter", _FakeWriter)
        monkeypatch.setattr(
            module,
            "build_trial_slope_stats_compatible_groups",
            lambda files: [group],
        )

        out_paths = module.main()
        assert len(out_paths) == 1
        assert "trial_slope_stats_group" in str(out_paths[0])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
