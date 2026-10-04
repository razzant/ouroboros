"""Deletion-first loop policy: manual control plus one measured Main pass."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import test_loop_compaction as _main_loop

_main_fit = _main_loop._fit
real_main_reclaim = _main_loop.real_main_reclaim  # the shared fixture: real Main/materializer boundary


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
        emit_progress=lambda _text, *, incident=None: None,
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

    source = Path("ouroboros/loop.py").read_text(encoding="utf-8")
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


# --- earlier capsules are re-folded only after the provider's own refusal ----------------------

def _long_earlier_capsule(run, monkeypatch):
    """Put an earlier capsule whose retelling is long enough to shrink again at messages[2]."""
    from copy import deepcopy
    from ouroboros import context_compaction as cc
    from tests.test_context_reclaim_materializer import _request, _unit

    stub = cc._call_summarizer
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **_kw: {
        part.source_id: "The earlier source was read closely and retold at length. " * 150 for part in parts})
    old = _unit("old-long", "old long evidence ")
    compacted, receipt, _usage = cc.compact_tool_history_llm(
        old, request=_request(old, 100_000), drive_root=run.root, task_id="seed-long",
        exposed_units=cc.exposed_context_units(old, old))
    monkeypatch.setattr(cc, "_call_summarizer", stub)
    assert receipt.status == "applied" and len(compacted) == 1
    capsule = deepcopy(compacted[0])
    run.context.messages[2] = capsule
    run.context.tools._ctx._last_context_observation = {
        "exposed_units": cc.exposed_context_units(run.context.messages, run.context.messages)}
    units = cc._atomic_units(run.context.messages)
    raw = next(unit for unit in units if not unit.generation)
    earlier = next(unit for unit in units if unit.generation)
    assert earlier.start == 2 and earlier.predicted_reclaim_tokens > 0 and raw.start > earlier.start
    return deepcopy(capsule), raw, earlier


@pytest.mark.parametrize("refused", [False, True])
def test_only_a_provider_refusal_refolds_earlier_capsules(real_main_reclaim, monkeypatch, refused):
    """Both directions: a pass the raw sources cannot satisfy keeps the earlier capsule
    unless the provider itself refused the request; after that typed refusal the capsule is
    re-folded as the last resort, after the raw unit although it stands earlier."""
    from ouroboros import context_compaction as cc, loop

    run = real_main_reclaim
    context = run.context
    capsule, raw, earlier = _long_earlier_capsule(run, monkeypatch)
    minimum_goal = raw.context_size_tokens + earlier.context_size_tokens + 25000
    disposition = _main_fit(action="send", profile="owner_low", mode="low", goal=0,
                       target_deficit=0, capacity_deficit=0)

    receipt = loop._run_main_reclaim(context, disposition, minimum_goal_tokens=minimum_goal,
                                     provider_refused=refused)

    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0
    assert not receipt.goal_reached
    assert receipt.selected_unit_ids == ((raw.unit_id, earlier.unit_id) if refused else (raw.unit_id,))
    assert {part.root_id for part in run.calls} == ({raw.unit_id, earlier.unit_id} if refused else {raw.unit_id})
    assert (context.messages[2] != capsule) is refused
    if refused:  # one host record for the uninterrupted range, in transcript order, one generation up
        record, meta = context.messages[2]["content"][0]["text"], cc._capsule_metadata(context.messages[2])[1]
        assert len(context.messages) == 3 and meta["generation"] == 2
        assert record.index(f"Source unit {earlier.unit_id}") < record.index(f"Source unit {raw.unit_id}")
        old = cc._capsule_metadata(capsule)[1]  # the earlier capsule's original provenance survives the re-fold
        assert set(old["source_hashes"]) < set(meta["source_hashes"])
        assert all(ref in meta["source_refs"] for ref in old["source_refs"])
    assert run.calls and run.events[-1]["deficit_tokens"] == 0
    assert run.events[-1]["reclaim_goal_tokens"] == minimum_goal


def test_refusal_takes_raw_sources_before_earlier_capsules(real_main_reclaim, monkeypatch):
    """After a refusal, a goal the raw sources reach leaves the earlier capsule alone."""
    from ouroboros import loop

    run = real_main_reclaim
    capsule, raw, _earlier = _long_earlier_capsule(run, monkeypatch)
    receipt = loop._run_main_reclaim(run.context, _main_fit(action="send", profile="owner_low", mode="low", goal=0,
                                                       target_deficit=0, capacity_deficit=0),
                                     minimum_goal_tokens=1, provider_refused=True)

    assert receipt.status == "applied" and receipt.selected_unit_ids == (raw.unit_id,)
    assert {part.root_id for part in run.calls} == {raw.unit_id}
    assert run.context.messages[2] == capsule
