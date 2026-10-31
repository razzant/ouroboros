"""Superseded host state leaves only where a request already departs from the last one (#2333).

A round that only appends keeps every row and its cached prefix; a tool load, a cold
resume or a rewrite re-reads from its departure, and superseded host rows there leave
with their exact bytes retained and one record naming them. How often a skill's
snapshot changes cannot multiply what the conversation holds.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import context_compaction  # noqa: F401 — public import establishes its leaf facade
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
from ouroboros.context_source_view import (HOST_STATE_RETIRED_NAME, _obsolete_host_rows,
                                           retire_superseded_at_departure, superseded_host_fact)
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.send_clock import CLOCK_ROW_NAME
from ouroboros.transcript_prefix import departure, observe_send
from tests.test_skill_context_snapshot import _write, publisher  # noqa: F401 -- the fixture

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}]


def _transcript(rounds: int) -> list:
    rows = [{"role": "system", "content": "Stable governance"}, {"role": "user", "content": "Owner request"}]
    for index in range(rounds):
        rows += [{"role": "assistant", "content": f"step {index}"},
                 bound_host_row(CONTEXT_FACTS_NAME, f"[CONTEXT_FACTS] round {index}"),
                 bound_host_row(CLOCK_ROW_NAME, f"[Host clock at this request: {index}]")]
    return rows


def test_a_round_that_only_appends_keeps_every_row(tmp_path):
    slot = SimpleNamespace()
    messages = _transcript(3)
    observe_send(slot, messages, round_idx=3, tools=TOOLS)
    messages += [{"role": "assistant", "content": "step 3"}, bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 3")]
    assert departure(slot, messages, TOOLS) is None
    assert retire_superseded_at_departure(slot, messages, TOOLS, drive_root=tmp_path, task_id="t") == (messages, None)
    # Visible to the mind meanwhile; the facts row written now supersedes the latest one too.
    assert "superseded host rows 6 " in superseded_host_fact(messages)


def test_a_tool_load_retires_every_superseded_row_with_exact_custody(tmp_path):
    slot = SimpleNamespace()
    messages = _transcript(4)
    observe_send(slot, messages, round_idx=4, tools=TOOLS)
    loaded = [*TOOLS, {"type": "function", "function": {"name": "mcp_x__y", "description": "e", "parameters": {}}}]
    superseded = sorted(_obsolete_host_rows(messages))
    retired = [messages[i] for i in superseded]
    rebuilt, fact = retire_superseded_at_departure(slot, messages, loaded, drive_root=tmp_path, task_id="t")
    assert (fact["departure"], fact["cause"], fact["rows"]) == (0, "tools", 6)
    kinds = [row.get(HOST_CONTEXT_KIND_KEY) for row in rebuilt]
    assert kinds.count(CONTEXT_FACTS_NAME) == kinds.count(CLOCK_ROW_NAME) == 1  # the latest of each stays
    assert rebuilt[-2:] == messages[-2:] and kinds.count(HOST_STATE_RETIRED_NAME) == 1
    record = next(row for row in rebuilt if row.get(HOST_CONTEXT_KIND_KEY) == HOST_STATE_RETIRED_NAME)
    assert "6 superseded host rows" in record["content"] and "nothing summarized" in record["content"]
    stored = json.loads(read_actor_source_bytes(tmp_path, "t", fact["checkpoint_ref"]))
    assert stored["rows"] == retired and stored["positions"] == superseded  # exact bytes and places, before removal
    assert [row for row in rebuilt if row.get("role") == "assistant"] == [
        row for row in messages if row.get("role") == "assistant"]  # the work stays whole


def test_repeated_departures_keep_one_record_that_holds_the_earlier_ones(tmp_path):
    slot, messages, tools, refs = SimpleNamespace(), _transcript(3), list(TOOLS), []
    for load in range(3):
        observe_send(slot, messages, round_idx=load, tools=tools)
        messages = [*messages, {"role": "assistant", "content": f"after load {load}"},
                    bound_host_row(CONTEXT_FACTS_NAME, f"[CONTEXT_FACTS] after load {load}")]
        tools = [*tools, {"type": "function", "function": {"name": f"mcp_x__t{load}", "description": "e", "parameters": {}}}]
        messages, fact = retire_superseded_at_departure(slot, messages, tools, drive_root=tmp_path, task_id="t")
        records = [i for i, row in enumerate(messages) if row.get(HOST_CONTEXT_KIND_KEY) == HOST_STATE_RETIRED_NAME]
        assert fact is not None and len(records) == 1 and records[0] not in _obsolete_host_rows(messages)
        refs.append(fact["checkpoint_ref"])
    for earlier, later in zip(refs, refs[1:]):  # each record's exact bytes sit in the next one's rows
        rows = json.loads(read_actor_source_bytes(tmp_path, "t", later))["rows"]
        assert any(json.dumps(earlier, sort_keys=True) in str(row.get("content")) for row in rows)


def test_a_rewrite_retires_only_from_its_departure_on(tmp_path):
    slot = SimpleNamespace()
    messages = _transcript(4)
    observe_send(slot, messages, round_idx=4, tools=TOOLS)
    start = 8  # an edit at a later row: earlier superseded rows stay, and the cached part before it too
    messages[start] = {"role": "assistant", "content": "step 2, folded by the mind"}
    rebuilt, fact = retire_superseded_at_departure(slot, messages, TOOLS, drive_root=tmp_path, task_id="t")
    assert (fact["departure"], fact["cause"]) == (start, "messages")
    assert rebuilt[:start] == messages[:start]
    assert all(i < start for i in _obsolete_host_rows(rebuilt)
               if rebuilt[i].get(HOST_CONTEXT_KIND_KEY) != HOST_STATE_RETIRED_NAME)


def test_rows_stay_without_a_recorded_send_or_when_their_exact_bytes_cannot_be_retained(tmp_path, monkeypatch):
    from ouroboros import artifacts

    messages = _transcript(3)
    # No recorded send: nothing proves the provider's cached prefix is gone.
    assert retire_superseded_at_departure(SimpleNamespace(), messages, TOOLS, drive_root=tmp_path, task_id="t") == (
        messages, None)
    slot = SimpleNamespace()
    observe_send(slot, messages, round_idx=3, tools=TOOLS)
    loaded = [*TOOLS, {"type": "function", "function": {"name": "mcp_x__y", "description": "e", "parameters": {}}}]
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    assert retire_superseded_at_departure(slot, messages, loaded, drive_root=tmp_path, task_id="t") == (messages, None)


def test_a_rebuild_shows_the_current_snapshot_once_and_changes_cannot_multiply(publisher, monkeypatch):  # noqa: F811
    from ouroboros import peer_roster
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_round_compaction
    from ouroboros.skill_context_snapshot import SNAPSHOT_KIND, visible_snapshots

    root, skill, ctx, path = publisher
    monkeypatch.setattr(peer_roster, "maybe_append_roster_note", lambda *a: False)
    schemas = list(TOOLS)
    round_ctx = _CompactionRoundContext(tools=SimpleNamespace(_ctx=ctx), drive_root=root, drive_logs=root / "logs",
                                        task_id="root", round_idx=1, event_queue=None, emit_progress=lambda x: None,
                                        tool_schemas=schemas)
    messages = [{"role": "system", "content": "Stable governance"}, {"role": "user", "content": "Owner request"}]
    _write(path, "A")
    messages = _run_round_compaction(messages, round_ctx)
    for round_idx, text in enumerate(("B", "C", "D", "E"), start=2):
        observe_send(ctx, messages, round_idx=round_idx, tools=schemas)
        messages.append({"role": "assistant", "content": f"work {round_idx}"})
        _write(path, text)
        before = list(messages)
        messages = _run_round_compaction(messages, round_ctx)
        assert messages[:len(before)] == before  # an appending round never rewrites
    notes = [row for row in messages if row.get(HOST_CONTEXT_KIND_KEY) == SNAPSHOT_KIND]
    assert len(notes) == 2 and visible_snapshots(messages)[skill.name][1]["state"] == "outdated"
    # A tool load departs from the start: the outdated text leaves and the current one is shown in full.
    observe_send(ctx, messages, round_idx=6, tools=schemas)
    schemas.append({"type": "function", "function": {"name": "mcp_x__y", "description": "e", "parameters": {}}})
    messages = _run_round_compaction(messages, round_ctx)
    assert visible_snapshots(messages)[skill.name][1]["text"] == "E"
    assert any(row.get(HOST_CONTEXT_KIND_KEY) == HOST_STATE_RETIRED_NAME for row in messages)
    assert not any('"text": "A"' in str(row.get("content")) for row in messages)
