"""The acceptance source selector is an append-only transcript row (issue #906).

Provider prompt caches reuse a previous request only when that whole request is a
byte-prefix of the next one. Replacing the trailing observation row every round
made no request a prefix of its predecessor, so the cache never extended past the
first round. These tests pin the two halves of the fix: the rendered facts no
longer carry the per-round ``tool_count``, and an already-sent row is never
removed, rewritten, or merged into.
"""
from __future__ import annotations

import copy
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.loop_messages import (
    _append_or_merge_user_content,
    _record_owner_directive,
    prepare_acceptance_observation,
)


@pytest.fixture
def eligible(tmp_path):
    """Smallest root-task context that receives the selector: delivery control asks
    Main to select a completion, so the observation row is rendered."""
    ctx = SimpleNamespace(
        task_id="root", task_attempt=1, drive_root=tmp_path,
        task_metadata={"root_task_id": "root"}, task_contract={}, is_direct_chat=False,
        _loop_mailbox_seen_ids=set(), _owner_directives=[], _delivery_control_required=True,
    )
    _record_owner_directive(ctx, source="initial_user", content="Prepare the full report.", msg_id="first")
    messages = [{"role": "user", "content": "Prepare the full report."}]
    trace = {"tool_calls": [{"tool": "write_file", "is_error": False, "status": "ok", "args": {"path": "report.md"}},
                            {"tool": "read_file", "is_error": False, "status": "ok", "args": {}}],
             "reasoning_notes": []}
    return ctx, messages, trace


def _advance_round(messages: list, trace: dict, idx: int) -> None:
    """Simulate the next round: a tool call happened and the transcript grew."""
    messages.append({"role": "assistant", "content": f"Working, step {idx}.",
                     "tool_calls": [{"id": f"c{idx}", "type": "function",
                                     "function": {"name": "read_file", "arguments": "{}"}}]})
    messages.append({"role": "tool", "tool_call_id": f"c{idx}", "content": "file body"})
    trace["tool_calls"].append({"tool": "read_file", "is_error": False, "status": "ok", "args": {}})


def _observation_rows(messages: list) -> list:
    return [row for row in messages if row.get("acceptance_observation")]


def test_unchanged_owner_facts_keep_exactly_one_frozen_row(eligible):
    ctx, messages, trace = eligible

    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)
    rows = _observation_rows(messages)
    assert len(rows) == 1
    first_index = messages.index(rows[0])
    frozen = copy.deepcopy(rows[0])

    for idx in (1, 2):
        _advance_round(messages, trace, idx)
        prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)

    rows = _observation_rows(messages)
    assert len(rows) == 1, "an unchanged selector must not append a second row"
    assert messages.index(rows[0]) == first_index, "the sent row must stay where it was appended"
    assert rows[0] == frozen, "the sent row must stay byte-identical"


def test_rendered_facts_omit_the_per_round_tool_count(eligible):
    ctx, messages, trace = eligible

    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)

    note = _observation_rows(messages)[0]["content"]
    assert "tool_count" not in note
    payload = json.loads(note.split("\n")[1])
    assert "tool_count" not in payload
    assert payload["owner_source_sha256"] == ctx._acceptance_observation["owner_source_sha256"]
    # The stored observation still carries it: loop_delivery bounds
    # material_tool_indices with this count.
    assert ctx._acceptance_observation["tool_count"] == len(trace["tool_calls"])


def test_changed_owner_facts_append_a_second_row_and_keep_the_first(eligible):
    ctx, messages, trace = eligible

    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)
    first = copy.deepcopy(_observation_rows(messages)[0])

    _advance_round(messages, trace, 1)
    _record_owner_directive(ctx, source="direct_incoming", content="Also cover Q3.", msg_id="followup")
    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)

    rows = _observation_rows(messages)
    assert len(rows) == 2
    assert rows[0] == first, "the earlier sent row must not be rewritten"
    assert rows[1] is messages[-1], "the new selector is appended at the end"
    assert rows[1]["content"] != first["content"]


def test_fence_token_change_appends_a_second_row(eligible):
    ctx, messages, trace = eligible

    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)
    first = copy.deepcopy(_observation_rows(messages)[0])

    _advance_round(messages, trace, 1)
    ctx._task_acceptance_fence_token = "fence-1"
    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)

    rows = _observation_rows(messages)
    assert len(rows) == 2
    assert rows[0] == first
    assert '"fence_token": "fence-1"' in rows[1]["content"]


def test_merge_never_rewrites_a_sent_observation_row(eligible):
    ctx, messages, trace = eligible
    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)
    frozen = copy.deepcopy(messages[-1])

    _append_or_merge_user_content(messages, "note")

    assert messages[-2] == frozen
    assert messages[-1] == {"role": "user", "content": "note"}


def test_merge_still_merges_into_an_ordinary_user_row():
    from types import SimpleNamespace
    from ouroboros.transcript_prefix import observe_send

    slot = SimpleNamespace()
    messages = [{"role": "system", "content": "system"}]
    observe_send(slot, messages, round_idx=1)
    messages.append({"role": "user", "content": "first"})
    _append_or_merge_user_content(messages, "second", slot=slot)

    assert len(messages) == 2
    assert messages[1]["content"] == "first\n\n---\n\nsecond"


def test_every_round_extends_the_previous_request_as_a_prefix(eligible):
    ctx, messages, trace = eligible
    snapshots = []

    for idx in range(3):
        if idx:
            _advance_round(messages, trace, idx)
        prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)
        snapshots.append(copy.deepcopy(messages))

    for previous, following in zip(snapshots, snapshots[1:]):
        assert len(following) >= len(previous)
        for before, after in zip(previous, following):
            assert before == after, "a sent message changed; the prompt cache cannot extend"
