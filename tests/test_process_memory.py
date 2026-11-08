"""Tests for process memory infrastructure.

Covers:
- A preserved commit's failed post-commit tests stay a typed trace fact
- Post-task work stays off the reply critical path
"""

import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ─────────────────────────────────────────────────────────────────────────────
# A preserved commit's failed post-commit tests: a typed fact in the trace
# ─────────────────────────────────────────────────────────────────────────────

class TestPostCommitTestFact:
    """The producer-to-trace half of the retired reflection trigger test survives on its own."""

    def test_a_preserved_commit_with_failing_post_commit_tests_keeps_a_typed_fact(self):
        """The ordinary self-modification commit reports failing post-commit tests
        as a WARNING appended to a success: the commit is preserved and the call is
        ok on every axis, while the producer states the failure as a typed fact the
        trace keeps; a clean commit gains no such field. Driven through the real
        producer seam (tools/git.py) and trace projection, never a Git operation.
        """
        import types

        from ouroboros._outcome_tool_errors import _classify_tool_errors
        from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata
        from ouroboros.tools import git as git_tools
        from ouroboros.tools.tool_result import (
            _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
        )

        def _row(test_warning: str) -> dict:
            ctx, sentinel = types.SimpleNamespace(), object()
            token = _install_tool_result_sidecar(ctx, sentinel)
            expected = "OK: committed to dev: fix parser[pushed: abc1234]" + test_warning
            try:
                text = git_tools._publish_post_commit_test_fact(ctx, expected, test_warning)
                published = _published_tool_result(ctx, sentinel)
            finally:
                _restore_tool_result_sidecar(token)
            assert text == expected  # the publication rule: the text stays byte-identical
            is_error = _typed_execution_failure(True, published) if published else False
            return {"tool": "commit_reviewed", "result": text, "is_error": is_error,
                    **_typed_result_metadata("commit_reviewed", text, is_error, published)}

        failed = _row("\n\n⚠️ TESTS_FAILED (commit preserved): post-commit verification failed.")
        assert failed["is_error"] is False and failed["status"] == "ok"
        assert not any(_classify_tool_errors({"tool_calls": [failed]}).values())  # no outcome reader counts it
        assert failed["post_commit_tests"] == "failed"
        assert "post_commit_tests" not in _row("")


class TestEmitTaskResultsPostTaskWorkNotOnCriticalPath:
    """Regression guard for the v4.39.0 UX regression: paid post-task work must
    not run on the reply critical path. The synchronous variant added a 1–3 s
    LLM round before `send_message` was dispatched; the async daemon-thread
    variant keeps reply latency low.
    """

    def _make_minimal_env(self, tmp_path):
        import pathlib

        class FakeEnv:
            drive_root = tmp_path
            repo_dir = str(tmp_path)

            def drive_path(self, sub):
                return pathlib.Path(self.drive_root) / sub

        return FakeEnv()

    def test_post_task_work_is_not_called_synchronously_in_emit_task_results(self, tmp_path):
        """emit_task_results must NOT run a paid post-task stage inline — it
        belongs inside the daemon thread started by
        `_run_post_task_processing_async`."""
        import unittest.mock as mock
        from ouroboros.agent_task_pipeline import emit_task_results

        env = self._make_minimal_env(tmp_path)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        task = {"id": "t1", "type": "task", "text": "goal", "chat_id": 1}
        usage = {"rounds": 5, "cost": 1.0, "prompt_tokens": 0, "completion_tokens": 0}
        llm_trace = {"tool_calls": []}

        with mock.patch("ouroboros.agent_task_pipeline._run_memory_fallback_draft") as mock_draft, \
             mock.patch("ouroboros.agent_task_pipeline._run_post_task_processing_async") as mock_async, \
             mock.patch("ouroboros.agent_task_pipeline._store_task_result"), \
             mock.patch("ouroboros.review_evidence.collect_review_evidence",
                        return_value={}, create=True):
            pending_events: list = []
            emit_task_results(
                env=env, memory=mock.MagicMock(), llm=mock.MagicMock(),
                pending_events=pending_events,
                task=task, text="reply",
                usage=usage, llm_trace=llm_trace,
                start_time=0.0,
                drive_logs=tmp_path / "logs",
            )

        # Paid stages are the responsibility of the async helper —
        # emit_task_results must not call them directly. Otherwise reply
        # latency regresses.
        mock_draft.assert_not_called()
        # The async helper must still be invoked exactly once.
        mock_async.assert_called_once()
