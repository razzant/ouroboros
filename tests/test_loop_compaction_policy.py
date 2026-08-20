"""Deletion-first loop policy: manual control plus one measured Main pass."""

from __future__ import annotations

from types import SimpleNamespace

def _ctx(tmp_path, *, pending=None):
    from ouroboros import loop

    inner = SimpleNamespace(_pending_compaction=pending)
    return loop._CompactionRoundContext(
        tools=SimpleNamespace(_ctx=inner),
        drive_root=tmp_path,
        drive_logs=tmp_path / "logs",
        task_id="task",
        round_idx=77,
        event_queue=None,
        emit_progress=lambda _text: None,
    )


def test_no_manual_request_is_byte_identical_at_every_round(tmp_path, monkeypatch):
    from ouroboros import loop

    called = False

    def forbidden(*_a, **_kw):
        nonlocal called
        called = True
        raise AssertionError("fixed/routine compaction must not run")

    monkeypatch.setattr(loop, "compact_tool_history_llm", forbidden)
    messages = [{"role": "user", "content": "x" * 1_500_000}]
    result, usage = loop._run_round_compaction(
        messages,
        _ctx(tmp_path),
    )
    assert result is messages
    assert usage is None
    assert called is False


def test_manual_request_uses_the_shared_typed_materializer(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.context_budget import ContextReclaimReceipt

    seen = {}
    rebuilt = [{"role": "assistant", "content": "summary"}]
    receipt = ContextReclaimReceipt(
        status="applied",
        before_transcript_sha256="a" * 64,
        after_transcript_sha256="b" * 64,
        selection_fingerprint="c" * 64,
        selected_unit_ids=("unit",),
        reclaimed_tokens=10,
        goal_reached=False,
        checkpoint_ref={"path": "checkpoint"},
        capsule_refs=(),
    )

    def fake(messages, **kwargs):
        seen.update(kwargs)
        return rebuilt, receipt, {"prompt_tokens": 3, "completion_tokens": 2}

    monkeypatch.setattr(loop, "compact_tool_history_llm", fake)
    context = _ctx(tmp_path, pending=4)
    result, usage = loop._run_round_compaction([{"role": "user", "content": "go"}], context)
    assert result is rebuilt
    assert usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert seen["keep_recent"] == 4
    assert seen["negative_memo"] is context.tools._ctx._context_reclaim_negative_memo
    assert context.tools._ctx._pending_compaction is None


def test_old_main_trigger_authorities_are_deleted():
    from pathlib import Path

    # v7 L-B split: the negative sweeps the whole loop family so no leaf
    # revives a deleted trigger authority.
    source = "".join(
        path.read_text(encoding="utf-8")
        for path in [Path("ouroboros/loop.py"), *sorted(Path("ouroboros").glob("loop_*.py"))]
    )
    budget = Path("ouroboros/context_budget.py").read_text(encoding="utf-8")
    for symbol in (
        "EMERGENCY_COMPACTION_CHARS",
        "LOW_EMERGENCY_COMPACTION_CHARS",
        "COMPACTION_HYSTERESIS_REGION_GROWTH",
        "COMPACTION_HYSTERESIS_ROUNDS",
        "_compaction_floor_chars",
        "_emergency_keep_recent",
    ):
        assert symbol not in source
        assert symbol not in budget
    assert "round_idx > 6" not in source
