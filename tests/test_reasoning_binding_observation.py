"""Retirement is one cache-break fact, one updating task row and a dated model fact."""

import copy
import json
import logging
import queue
from types import SimpleNamespace

import pytest

from ouroboros import loop, task_pacing, transcript_prefix
from ouroboros.loop_messages import append_context_facts
from ouroboros.tools.tool_context import ToolContext
from tests._dialogue import progress_rows
from tests.test_reasoning_binding import TOOLS, answered, history, turn
from tests.test_round_facts_line import _facts_ctx


def _events(root):
    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _main_context(slot, messages, usage, round_idx):
    return SimpleNamespace(tools=SimpleNamespace(_ctx=slot), messages=messages, tool_schemas=TOOLS,
                           accumulated_usage=usage, round_idx=round_idx, task_id=slot.task_id,
                           drive_logs=slot.drive_root / "logs", event_queue=getattr(slot, "event_queue", None))


@pytest.mark.parametrize("baseline", ["answered", "restored", "absent"])
def test_retirement_is_one_checkpoint_even_when_the_answered_observer_has_no_baseline(tmp_path, caplog, baseline):
    messages = history()
    slot = answered(messages) if baseline == "answered" else SimpleNamespace()
    if baseline == "restored":
        slot._last_context_observation = {"messages": copy.deepcopy(messages), "tool_schemas": TOOLS}
    slot.task_id, slot.drive_root = "retirement-task", tmp_path
    slot.event_queue = queue.Queue()
    progress, usage = [], {}
    slot._reasoning_retirement_progress = lambda text, **row: progress.append((text, row))
    messages[0]["content"] = "Changed system."
    ctx = _main_context(slot, messages, usage, 2)
    with caplog.at_level(logging.WARNING, logger="ouroboros.transcript_prefix"):
        fact = transcript_prefix.retire_main_reasoning(ctx)
        assert transcript_prefix.retire_main_reasoning(ctx) is None
    assert len(caplog.records) == 1
    assert fact == {"index": 0, "cause": "no_baseline" if baseline == "absent" else "system", "turns": 3, "blocks": 3}
    # Use the real answered boundary: it persists the restored source and passes tools.
    loop._record_transcript_prefix(slot, messages, 2, usage, slot.event_queue, slot.task_id, tmp_path / "logs", TOOLS)
    assert transcript_prefix.tools_digest(TOOLS) == getattr(slot, transcript_prefix.TOOLS_DIGEST_ATTR)
    [checkpoint] = _events(tmp_path)
    assert checkpoint["checkpoint_kind"] == "prompt_prefix_break" and checkpoint["reasoning_retired"] == fact
    assert checkpoint["progress_row_emitted"] is True
    assert usage["reasoning_retired"] == usage["prompt_prefix_breaks"] == 1
    assert usage["reasoning_retired_turns"] == usage["reasoning_retired_blocks"] == 3
    assert progress[0][1] == {"card_row": "timeline", "card_row_id": "retirement-task:reasoning_retired"}
    assert len(progress) == slot.event_queue.qsize() == 1
    assert not progress_rows(tmp_path), "Main's progress callback already owns the visible row"


@pytest.mark.parametrize("sanctioned", [False, True])
def test_each_rewrite_updates_the_same_task_row_and_retains_the_sanction(tmp_path, sanctioned):
    messages, usage, progress = history(), {}, []
    slot = answered(messages)
    slot.task_id, slot.drive_root = "retirement-task", tmp_path
    slot._reasoning_retirement_progress = lambda text, **row: progress.append((text, row))
    for round_idx in (2, 3):
        if sanctioned:
            transcript_prefix.sanction_rewrite(slot, "compaction")
        messages[3]["content"] = f"Revision {round_idx}."
        fact = transcript_prefix.retire_main_reasoning(_main_context(slot, messages, usage, round_idx))
        assert fact["turns"] == (2 if round_idx == 2 else 1)
        loop._record_transcript_prefix(slot, messages, round_idx, usage, None, slot.task_id, tmp_path / "logs", TOOLS)
        messages.extend(turn(f"c{round_idx + 2}"))
    assert len(_events(tmp_path)) == len(progress) == 2
    assert progress[0][1] == progress[1][1]
    assert "1 history change (2 turns)" in progress[0][0]
    assert "2 history changes (3 turns)" in progress[1][0]
    assert all("blocks" not in text for text, _ in progress)
    assert usage["reasoning_retired"] == 2
    assert usage["prompt_prefix_breaks"] == 2  # a sanction names the change, it never exempts it
    assert all(row["sanctioned_by"] == ("compaction" if sanctioned else None) for row in _events(tmp_path))


def test_append_only_send_is_silent_and_helper_retirement_is_logs_only(tmp_path, caplog):
    messages, usage = history(), {}
    slot = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="helper-task", event_queue=queue.Queue())
    transcript_prefix.observe_send(slot, messages, tools=TOOLS, round_idx=1)
    progress = []
    slot._reasoning_retirement_progress = lambda text, **row: progress.append((text, row))
    messages.extend(turn("c4"))
    assert transcript_prefix.prepare_reasoning_for_send(slot, messages, TOOLS, round_idx=2, usage=usage) is None
    assert usage == {} and slot.event_queue.empty() and _events(tmp_path) == []
    assert not caplog.records
    messages[3]["content"] = "Changed result."
    transcript_prefix.prepare_reasoning_for_send(slot, messages, TOOLS, round_idx=2, usage=usage)
    [checkpoint] = _events(tmp_path)
    assert checkpoint == slot.event_queue.get_nowait()["data"]
    assert checkpoint["reasoning_retired"]["turns"] == 3
    assert not progress_rows(tmp_path) and not progress
    assert not {"system_type", "text", "card_row", "card_row_id", "progress_row_emitted",
                "toast_once", "incident_id", "incident_code", "health"} & checkpoint.keys()
    assert slot.event_queue.empty()


def test_retirement_identity_refreshes_same_round_facts_and_expires_next_round(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    ctx = _facts_ctx(tmp_path, monkeypatch)
    assert append_context_facts(ctx)
    original = ctx.messages[-1]["content"]
    assert "Earlier reasoning" not in original
    # The transcript/route is unchanged: only retirement identity invalidates the row key.
    ctx.accumulated_usage["_reasoning_retired_round"] = {
        "round": ctx.round_idx, "index": 3, "cause": "messages", "turns": 2, "blocks": 2,
    }
    assert append_context_facts(ctx)
    current = ctx.messages[-1]["content"]
    assert current.count("Earlier reasoning not replayed") == 1
    assert "an earlier part of this conversation changed; your signed reasoning from 2 later turns" in current
    assert "message 3" not in current and "(messages)" not in current
    assert "your visible text, tool calls and results are unchanged." in current
    assert not append_context_facts(ctx)
    ctx.round_idx += 1
    assert append_context_facts(ctx)
    assert "Earlier reasoning" not in ctx.messages[-1]["content"]
    assert any(row.get("content") == current for row in ctx.messages), "sent facts remain historical"


@pytest.mark.parametrize("turns", [1, 2])
@pytest.mark.parametrize("cause,sanction,explanation", [
    ("messages", "compaction", "context folded"),
    ("system", None, "the system view was rebuilt"),
    ("tools", None, "the tool set changed"),
    ("no_baseline", None, "the task resumed without a record of its last request"),
])
def test_model_fact_explains_the_change_without_indices_or_internal_cause_codes(cause, sanction, explanation, turns):
    usage = {"_reasoning_retired_round": {"round": 4, "index": 123, "cause": cause, "turns": turns,
                                          "blocks": 3, "sanctioned_by": sanction}}
    text = transcript_prefix.reasoning_retirement_fact(usage, 4)
    assert f"conversation changed ({explanation}); your signed reasoning from {turns} later turn" in text
    assert f"later {'turn' if turns == 1 else 'turns'} was removed" in text
    assert "123" not in text and f"({cause})" not in text and "blocks" not in text
    assert not transcript_prefix.reasoning_retirement_fact(usage, 5)


@pytest.mark.parametrize("change", ["image", "blocks", "tools", "cache_only"])
def test_cache_observer_sees_images_blocks_and_schemas_but_not_cache_markers(change):
    messages = [{"role": "system", "content": "go"}, {"role": "user", "content": [
        {"type": "text", "text": "ab"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
    ]}]
    schemas, slot = copy.deepcopy(TOOLS), answered(messages)
    if change == "image":
        messages[1]["content"][1]["image_url"]["url"] = "data:image/png;base64,Yg=="
    elif change == "blocks":
        messages[1]["content"][:1] = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    elif change == "tools":
        schemas[0]["function"]["description"] = "Changed contract."
    else:
        messages[1]["content"][0]["cache_control"] = {"type": "ephemeral"}
        schemas[0]["cache_control"] = {"type": "ephemeral"}
    fact = transcript_prefix.observe_send(slot, messages, tools=schemas, round_idx=2)
    if change == "cache_only":
        assert fact is None
    else:
        assert fact["index"] == (0 if change == "tools" else 1)
        assert fact["kind"] == ("tools_changed" if change == "tools" else "tail_replaced")


@pytest.mark.parametrize("signed_reasoning", [True, False])
def test_a_helper_send_counts_its_change_once_signed_or_not(tmp_path, signed_reasoning):
    """Native review and consolidation count a change as Main does: retirement and the
    send observer share one identity, and a model without signed reasoning still counts."""
    messages = history()
    if not signed_reasoning:
        for message in messages:
            message.pop("reasoning_details", None)
    slot = answered(messages)
    slot.task_id, slot.drive_root, usage = "helper-task", tmp_path, {}
    transcript_prefix.sanction_rewrite(slot, "compaction")
    messages[3]["content"] = "folded tool result"
    transcript_prefix.prepare_reasoning_for_send(slot, messages, TOOLS, round_idx=2, usage=usage)
    transcript_prefix.count_accepted_send(slot, messages, TOOLS, round_idx=2, usage=usage)
    assert usage["prompt_prefix_breaks"] == 1 and len(_events(tmp_path)) == 1
    assert [(b["round"], b["index"], b["sanctioned_by"]) for b in usage["_recent_prefix_breaks"]] == [(2, 3, "compaction")]
    record = json.loads(transcript_prefix.history_changes_path(tmp_path).read_text())
    assert record["origins"]["compaction"]["count"] == 1
    transcript_prefix.count_accepted_send(slot, [*messages, {"role": "user", "content": "more"}], TOOLS, round_idx=3, usage=usage)
    assert usage["prompt_prefix_breaks"] == 1  # an append is no change


def test_a_restored_slot_counts_its_first_send_against_the_observation(tmp_path):
    """A cold continuation keeps its context observation, not the send digests: the first
    send after the restore still compares with what was sent before it."""
    from ouroboros import owner_wait
    from ouroboros.context_budget import LIVE_IMAGE_BLOCKS_HIGH_WATER
    from ouroboros.loop_messages import _append_or_merge_user_content
    from ouroboros.tools.compact_context import record_context_view

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    sent = [{"role": "system", "content": "system"},
            {"role": "user", "content": [copy.deepcopy(image) for _ in range(LIVE_IMAGE_BLOCKS_HIGH_WATER)]}]
    previous = SimpleNamespace(task_id="cold-image", drive_root=tmp_path)
    record_context_view(previous, sent, [])
    transcript_prefix.observe_send(previous, sent, tools=[], round_idx=1)
    state = json.loads(json.dumps(owner_wait.continuation_state(previous, sent, {}, {}, 1, [], set())))
    for evict in (False, True):
        slot, messages, usage = SimpleNamespace(task_id="cold-image", drive_root=tmp_path), [], {}
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=slot), state, messages, {}, usage, set())
        _append_or_merge_user_content(messages, [copy.deepcopy(image)] if evict else "continue", slot=slot)
        assert transcript_prefix.prepare_reasoning_for_send(slot, messages, [], round_idx=2, usage=usage) is None
        loop._record_transcript_prefix(slot, messages, 2, usage, None, slot.task_id, tmp_path / "logs", [])
        assert usage.get("prompt_prefix_breaks", 0) == (1 if evict else 0)
        assert [b["sanctioned_by"] for b in usage.get("_recent_prefix_breaks", [])] == (["image_eviction"] if evict else [])
