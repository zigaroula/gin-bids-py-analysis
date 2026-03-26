"""Smoke tests for ComputeWorker — signal emission via mock processor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gin_bids_py_analysis.visualization.trial_stats.worker import (
    ComputeAllWorker,
    ComputeWorker,
    GroupComputeWorker,
)


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


class TestComputeAllWorker:
    def _make_subject_groups(self, synthetic_result):
        return {
            "01": synthetic_result.source_group,
            "02": synthetic_result.source_group,
        }

    def test_subject_done_emitted_for_each_subject(self, qtbot, synthetic_result):
        subject_groups = self._make_subject_groups(synthetic_result)
        factory = MagicMock(return_value=MagicMock(
            process_group=MagicMock(return_value=synthetic_result)
        ))
        params = MagicMock()

        worker = ComputeAllWorker(factory, subject_groups, params)
        received: list[str] = []

        def on_subject_done(subject_id, result):
            received.append(subject_id)

        worker.subject_done.connect(on_subject_done)

        with qtbot.waitSignal(worker.all_done, timeout=5000):
            worker.start()

        assert set(received) == {"01", "02"}

    def test_all_done_emitted_with_full_results(self, qtbot, synthetic_result):
        subject_groups = self._make_subject_groups(synthetic_result)
        factory = MagicMock(return_value=MagicMock(
            process_group=MagicMock(return_value=synthetic_result)
        ))
        params = MagicMock()

        worker = ComputeAllWorker(factory, subject_groups, params)

        with qtbot.waitSignal(worker.all_done, timeout=5000) as blocker:
            worker.start()

        all_results = blocker.args[0]
        assert set(all_results.keys()) == {"01", "02"}

    def test_error_signal_on_failure(self, qtbot, synthetic_result):
        subject_groups = {"01": synthetic_result.source_group}
        factory = MagicMock(return_value=MagicMock(
            process_group=MagicMock(side_effect=RuntimeError("bad"))
        ))
        params = MagicMock()

        worker = ComputeAllWorker(factory, subject_groups, params)

        with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
            worker.start()

        assert "bad" in blocker.args[0]

    def test_factory_called_once_per_subject(self, qtbot, synthetic_result):
        subject_groups = self._make_subject_groups(synthetic_result)
        mock_processor = MagicMock(
            process_group=MagicMock(return_value=synthetic_result)
        )
        factory = MagicMock(return_value=mock_processor)
        params = MagicMock()

        worker = ComputeAllWorker(factory, subject_groups, params)
        with qtbot.waitSignal(worker.all_done, timeout=5000):
            worker.start()

        assert factory.call_count == 2


class TestGroupComputeWorker:
    def test_error_signal_on_bridge_failure(self, qtbot, synthetic_result):
        """GroupComputeWorker emits error if build_group_file_group_from_results raises."""
        from unittest.mock import patch

        group_params = MagicMock()
        group_params.source_metric = "t_values"
        all_results = {"01": synthetic_result}

        worker = GroupComputeWorker(all_results, group_params)

        with patch(
            "gin_bids_py_analysis.visualization.trial_stats._bridge"
            ".build_trial_stats_compatible_groups",
            side_effect=RuntimeError("bridge error"),
        ):
            with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
                worker.start()

        assert "bridge error" in blocker.args[0]
