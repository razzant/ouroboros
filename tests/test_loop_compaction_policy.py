"""Deletion-first loop policy: the round rewrites nothing unless the mind asked for a view."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _ctx(tmp_path, *, pending=None, progress=None):
    from ouroboros import loop

    inner = SimpleNamespace(_pending_compaction=pending)
    return loop._CompactionRoundContext(
        tools=SimpleNamespace(_ctx=inner),
        drive_root=tmp_path,
        drive_logs=tmp_path / "logs",
        task_id="task",
        round_idx=77,
        event_queue=None,
        emit_progress=lambda text, *, incident=None: (progress.append(text) if progress is not None else None),
    )


def test_no_manual_request_is_byte_identical_at_every_round(tmp_path, monkeypatch):
    from ouroboros import context_compaction as cc, loop

    def forbidden(*_a, **_kw):
        raise AssertionError("fixed/routine compaction must not run")

    monkeypatch.setattr(cc, "compact_tool_history_llm", forbidden)
    messages = [{"role": "user", "content": "x" * 1_500_000}]
    result = loop._run_round_compaction(messages, _ctx(tmp_path))
    assert result is messages


def test_a_saved_count_only_request_is_dropped_and_disclosed_without_a_materializer(tmp_path, monkeypatch):
    """Both directions: the count-only form an earlier version saved named a helper that no longer
    exists, so it is cleared and disclosed and no materializer runs; the mind's authored view
    (a dict) still reaches the authored path."""
    from ouroboros import context_compaction as cc, loop, loop_round_limits

    monkeypatch.setattr(cc, "compact_tool_history_llm",
                        lambda *_a, **_kw: pytest.fail("a count-only request must not reach the materializer"))
    progress: list[str] = []
    context = _ctx(tmp_path, pending=4, progress=progress)
    messages = [{"role": "user", "content": "go"}]
    result = loop._run_round_compaction(messages, context)
    assert result is messages and context.tools._ctx._pending_compaction is None
    assert len(progress) == 1 and "working_note" in progress[0]

    seen = []
    monkeypatch.setattr(loop_round_limits, "_run_authored_context_view",
                        lambda messages, ctx, pending, names: seen.append(pending) or messages)
    authored = {"observed": {}, "working_note": "My own account."}
    context = _ctx(tmp_path, pending=authored, progress=progress)
    assert loop._run_round_compaction(messages, context) is messages
    assert seen == [authored] and len(progress) == 1
