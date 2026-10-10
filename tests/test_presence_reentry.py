"""Exact reentry facts through a real frozen Presence registry, without chat_history.

These are local consumer tests: real source store, task scope and tool dispatch;
no model, provider, HTTP process or Slack transport is involved.
"""

from dataclasses import replace
import json
import re

import pytest

from ouroboros import presence_continuation as pc
from ouroboros.task_results import write_task_result
from tests.test_presence_continuation_seams import KEY, _append, _binding, _row
from tests.test_presence_own_work import _ceiling, _registry


def _actor(root):
    ceiling = _ceiling()
    ceiling = replace(ceiling, tool_grants=tuple(grant for grant in ceiling.tool_grants
                                                if grant.name not in {"chat_history", "read_file"}))
    registry, ctx = _registry(root, ceiling, key=KEY, task_id="turn-me")
    write_task_result(root, ctx.task_id, "running", metadata=ctx.task_metadata)
    assert "chat_history" not in {item["function"]["name"] for item in registry.schemas()}
    return registry, ctx


def _reader(note):
    return json.loads(re.search(r'get_task_result\((\{[^\n]+?\})\)', note).group(1))


def _read_all(registry, arguments):
    projection = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    parts = []
    for start in range(0, projection["complete_chars"], 2000):
        page = json.loads(registry.execute("get_task_result", {
            **arguments, "source_start_char": start,
            "source_end_char": min(start + 2000, projection["complete_chars"]),
        }))["presence_reentry_source"]
        parts.append(page["text"])
        assert page["complete_sha256"] == arguments["presence_reentry_sha256"]
    return json.loads("".join(parts))


def test_large_single_message_is_fully_readable_inside_frozen_ceiling(tmp_path):
    registry, ctx = _actor(tmp_path)
    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    text = "start " + "full words\n" * 4000 + " THE LAST WORDS"
    _append(tmp_path, _row(KEY, text))
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert "every row" not in note and "1 earlier row(s)" in note
    assert "chat_history(" not in note and "THE LAST WORDS" not in note
    source = _read_all(registry, _reader(note))
    assert source["rows"][0]["text"] == text and source["gaps"] == []
    assert source["task_id"] == ctx.task_id and source["conversation_key"] == KEY
    assert "chat_history" not in {item["function"]["name"] for item in registry.schemas()}


def test_old_snapshot_remains_readable_after_new_reentry_and_rotation(tmp_path):
    from supervisor.state import rotate_chat_log_if_needed

    registry, ctx = _actor(tmp_path)
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, *[_row(KEY, f"row {i} " + "x" * 1000) for i in range(40)])
    old = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    new_cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "new boundary"))
    new = _reader(pc.reentry_note(_binding(tmp_path, new_cursor), ctx))
    assert old != new
    assert len(_read_all(registry, old)["rows"]) == 40
    assert _read_all(registry, new)["rows"][0]["text"] == "new boundary"


def test_reentry_reader_preserves_task_scope_and_integrity(tmp_path):
    registry, ctx = _actor(tmp_path)
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "private conversation facts"))
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    stranger, _ = _registry(tmp_path, binding="b" * 32, task_id="stranger")
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", arguments)
    invalid = json.loads(registry.execute("get_task_result", {
        **arguments, "source_start_char": -1, "source_end_char": 20,
    }))["presence_reentry_source"]
    assert invalid["reason"] == "source_range_invalid" and "text" not in invalid
    from ouroboros.artifacts import task_artifact_dir_path

    path = task_artifact_dir_path(tmp_path, ctx.task_id, create=False) / (
        f"source_handles/context_checkpoints/presence-reentry-{arguments['presence_reentry_sha256']}.json")
    path.write_text("tampered", encoding="utf-8")
    bad = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    assert bad["reason"] == "source_identity_mismatch" and "text" not in bad


@pytest.mark.parametrize("failure", ["reader_absent", "write_failed", "readback_failed"])
def test_unavailable_source_keeps_whole_observed_text_inline(tmp_path, monkeypatch, failure):
    from ouroboros.presence_authority import presence_ceiling_payload

    registry, ctx = _actor(tmp_path)
    if failure == "reader_absent":
        ceiling = _ceiling()
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(replace(ceiling, tool_grants=()))}
    else:
        def fail(*args, **kwargs):
            raise OSError("source unavailable")
        target = "store_actor_source_bytes" if failure == "write_failed" else "read_actor_source_bytes"
        monkeypatch.setattr(f"ouroboros.artifacts.{target}", fail)
    cursor = pc._chat_cursor(tmp_path)
    text = "uncut " + "q" * 30_000 + " END"
    monkeypatch.setattr(pc, "_REENTRY_SCAN_BYTES", 1000)
    _append(tmp_path, *[_row("telegram:bot-1:elsewhere:0", "foreign " * 1000) for _ in range(3)])
    _append(tmp_path, _row(KEY, text))
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert json.dumps(text) in note and "get_task_result(" not in note
    assert "Coverage: every row" in note


@pytest.mark.parametrize("failure", ["reader_absent", "write_failed", "readback_failed"])
def test_unavailable_source_after_scan_budget_names_chain_gap_offsets(tmp_path, monkeypatch, failure):
    from ouroboros.presence_authority import presence_ceiling_payload
    from supervisor.state import rotate_chat_log_if_needed

    _, ctx = _actor(tmp_path)
    if failure == "reader_absent":
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(
            replace(_ceiling(), tool_grants=()))}
    else:
        def fail(*args, **kwargs):
            raise OSError("source unavailable")
        target = "store_actor_source_bytes" if failure == "write_failed" else "read_actor_source_bytes"
        monkeypatch.setattr(f"ouroboros.artifacts.{target}", fail)
    archived = _append(tmp_path, _row(KEY, "archived before park"))
    archive_bytes = archived.stat().st_size
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    _append(tmp_path, _row(KEY, "before park"))
    cursor = pc._chat_cursor(tmp_path)
    live = _append(tmp_path, _row("telegram:bot-1:elsewhere:0", "foreign " * 100),
                   _row(KEY, "after scan budget"))
    physical_offset = live.stat().st_size
    bad_rows = [(b"{broken\n", "a malformed line"), (b"[1,2]\n", "a non-object line"),
                (b'{"partial":', "a row still being written")]
    _append(tmp_path, raw=b"".join(raw for raw, _ in bad_rows))
    monkeypatch.setattr(pc, "_REENTRY_SCAN_BYTES", 1)
    monkeypatch.setattr(pc, "_REENTRY_PAGE_BYTES", 20)
    assert pc._conversation_rows_since(tmp_path, cursor, KEY)[1][-1]["kind"] == "scan_budget_exhausted"

    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)

    assert '"after scan budget"' in note and "foreign " not in note
    assert "get_task_result(" not in note and "chat_history(" not in note
    assert "Coverage gaps" in note and "Coverage: every row" not in note
    assert "{path}" not in note and "{offset}" not in note
    for raw, description in bad_rows:
        # A preceding archive makes chain offsets differ from live-file offsets.
        assert f"{description} at the captured chat.jsonl chain byte {archive_bytes + physical_offset}" in note
        physical_offset += len(raw)


def _append_gap_locations(root, count=12):
    _append(root, _row(KEY, "before the gap interval"))
    cursor = pc._chat_cursor(root)
    offset, offsets, raw = cursor["offset"], [], b""
    for index in range(count):
        line = f"malformed gap {index}\n".encode()
        offsets.append(offset)
        raw += line
        offset += len(line)
    _append(root, raw=raw)
    return cursor, offsets


@pytest.mark.parametrize("failure", ["reader_absent", "write_failed"])
def test_unavailable_source_keeps_every_gap_location_inline(tmp_path, monkeypatch, failure):
    from ouroboros.presence_authority import presence_ceiling_payload

    _, ctx = _actor(tmp_path)
    if failure == "reader_absent":
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(
            replace(_ceiling(), tool_grants=()))}
    else:
        def fail(*args, **kwargs):
            raise OSError("source unavailable")
        monkeypatch.setattr("ouroboros.artifacts.store_actor_source_bytes", fail)
    cursor, offsets = _append_gap_locations(tmp_path)
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert "get_task_result(" not in note and "chat_history(" not in note
    assert all(f"a malformed line at logs/chat.jsonl byte {offset}" in note for offset in offsets)
    assert "- and 4 more" not in note


def test_retained_source_keeps_all_gap_locations_when_note_is_bounded(tmp_path):
    registry, ctx = _actor(tmp_path)
    cursor, offsets = _append_gap_locations(tmp_path)
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert "- and 4 more" in note and "chat_history(" not in note
    assert "positions named above" not in note
    assert "the retained reentry source lists every observed gap" in note
    source = _read_all(registry, _reader(note))
    assert [gap["offset"] for gap in source["gaps"]] == offsets
    assert all(gap["path"] == "logs/chat.jsonl" for gap in source["gaps"])
