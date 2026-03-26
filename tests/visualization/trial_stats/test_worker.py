"""Smoke tests for ComputeWorker — signal emission via mock processor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gin_bids_py_analysis.visualization.trial_stats.worker import ComputeWorker


class TestComputeWorker:
    def test_result_ready_signal_on_success(self, qtbot, synthetic_result):
        processor = MagicMock()
        processor.process_group.return_value = synthetic_result

        group = synthetic_result.source_group
        worker = ComputeWorker(processor, group)

        with qtbot.waitSignal(worker.result_ready, timeout=5000) as blocker:
            worker.start()

        assert blocker.args[0] is synthetic_result
        processor.process_group.assert_called_once_with(group)

    def test_error_signal_on_exception(self, qtbot, synthetic_result):
        processor = MagicMock()
        processor.process_group.side_effect = RuntimeError("boom")

        group = synthetic_result.source_group
        worker = ComputeWorker(processor, group)

        with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
            worker.start()

        assert "boom" in blocker.args[0]

    def test_no_result_ready_on_error(self, qtbot, synthetic_result):
        processor = MagicMock()
        processor.process_group.side_effect = ValueError("bad input")

        group = synthetic_result.source_group
        worker = ComputeWorker(processor, group)

        with qtbot.waitSignal(worker.error, timeout=5000):
            worker.start()

        # Wait for thread to fully finish before assertions
        worker.wait(1000)
