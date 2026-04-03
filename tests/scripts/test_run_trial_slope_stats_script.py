from __future__ import annotations

import shutil
import uuid
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_case_dir(case_name: str) -> Path:
    root = REPO_ROOT / "tests" / "_script_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{case_name}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _load_script_module() -> object:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_trial_slope_stats.py"
    spec = spec_from_file_location("run_trial_slope_stats", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_trial_slope_stats_script_main_smoke(monkeypatch) -> None:
    case_dir = _make_case_dir("run_slope_script_smoke")
    try:
        module = _load_script_module()

        class _FakeDataset:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

        class _FakeProcessor:
            def __init__(self, _params, resolver=None) -> None:
                del resolver

            def run(self, groups, writer, n_jobs=1):
                del groups, writer, n_jobs
                return [
                    case_dir
                    / "derivatives"
                    / "trial_slope_stats"
                    / "sub-01"
                    / "ieeg"
                    / "sub-01_task-decid_desc-trialslopestats_stats.h5"
                ]

        class _FakeWriter:
            def __init__(self, _params) -> None:
                pass

        monkeypatch.setattr(module, "BIDSDataset", _FakeDataset)
        monkeypatch.setattr(module, "build_subject_groups", lambda *_args, **_kwargs: ["dummy"])
        monkeypatch.setattr(module, "TrialSlopeStatsProcessing", _FakeProcessor)
        monkeypatch.setattr(module, "TrialSlopeStatsProcessingWriter", _FakeWriter)

        out_paths = module.main()
        assert len(out_paths) == 1
        assert "trial_slope_stats" in str(out_paths[0])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
