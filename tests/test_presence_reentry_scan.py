"""Bounded canonical reentry pages through the actor's existing own-result reader.

Real registry, frozen ceiling, task scope, artifacts and shared chat-chain reads;
no model, HTTP Host or provider transport runs in this local consumer fixture.
"""

import json
import hashlib

import pytest

from ouroboros import presence_continuation as pc
from tests.test_presence_continuation_seams import KEY, OTHER, _append, _binding, _row
from tests.test_presence_reentry import _actor, _reader


def _page(registry, arguments, offset):
    return json.loads(registry.execute("get_task_result", {
        **arguments, "presence_reentry_offset": offset,
    }))["presence_reentry_source"]


def _pages(registry, arguments):
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    assert source["history_status"] == "available"
    offset, pages = source["history_start"], []
    while offset is not None:
        page = _page(registry, arguments, offset)
        assert page["status"] == "ok", page
        assert page["start_offset"] == offset and page["history_end"] == source["history_end"]
        pages.append(page)
        offset = page["next_offset"]
        assert offset is None or offset > page["start_offset"]
        assert len(pages) < 100, "a page did not advance through its physical interval"
    assert pages[-1]["interval_exhausted"]
    return pages


def test_matching_live_cursor_reads_new_rows_without_hashing_archive_headers(tmp_path, monkeypatch):
    from ouroboros import utils
    from supervisor.state import rotate_chat_log_if_needed

    _append(tmp_path, _row(KEY, "old archived row"))
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    assert list((tmp_path / "archive").glob("chat_*.jsonl"))
    live = _append(tmp_path, _row(KEY, "before this park"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(OTHER, "another conversation"), _row(KEY, "after this park"))
    hashed, signature = [], utils.jsonl_generation_signature

    def observed(path):
        hashed.append(path)
        return signature(path)

    monkeypatch.setattr(utils, "jsonl_generation_signature", observed)
    rows, gaps = pc._conversation_rows_since(tmp_path, cursor, KEY)
    assert [row["text"] for row in rows] == ["after this park"] and gaps == []
    assert hashed == [live]


def test_frozen_actor_can_read_past_scan_budget_without_chat_history(tmp_path, monkeypatch):
    from ouroboros.jsonl_tail import JsonlChainSnapshot

    registry, ctx = _actor(tmp_path)
    _append(tmp_path, _row(KEY, "before this park"))
    cursor = pc._chat_cursor(tmp_path)
    monkeypatch.setattr(pc, "_REENTRY_SCAN_BYTES", 400)
    monkeypatch.setattr(pc, "_REENTRY_PAGE_BYTES", 512)
    wanted = ["first after foreign bytes", "second after foreign bytes"]
    _append(tmp_path, *[_row(OTHER, f"private other thread {i} " + "x" * 300) for i in range(12)],
            *[_row(KEY, text) for text in wanted])
    note = pc.reentry_note(_binding(tmp_path, cursor), ctx)
    assert "after the scan budget" in note and "chat_history(" not in note
    arguments = _reader(note)
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    _append(tmp_path, _row(KEY, "after the frozen observation"))
    read_sizes, read = [], JsonlChainSnapshot._read

    def observed(self, start, end):
        read_sizes.append(end - start)
        return read(self, start, end)

    monkeypatch.setattr(JsonlChainSnapshot, "_read", observed)
    pages = _pages(registry, arguments)
    assert [row["text"] for page in pages for row in page["rows"]] == wanted
    assert len(pages) > 2 and all(not page["gaps"] for page in pages)
    assert pages[-1]["end_offset"] == source["history_end"]
    assert read_sizes and max(read_sizes) <= 512  # physical reads remain bounded, including foreign rows
    assert "private other thread" not in json.dumps(pages)
    assert "chat_history" not in {item["function"]["name"] for item in registry.schemas()}


def test_tiny_pages_keep_one_oversized_row_whole_and_survive_rotation(tmp_path, monkeypatch):
    from supervisor.state import rotate_chat_log_if_needed

    registry, ctx = _actor(tmp_path)
    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    long_text = "full large row " + "unicode λ\n" * 1000 + " LAST WORDS"
    _append(tmp_path, _row(KEY, long_text), _row(KEY, "next complete row"))
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    monkeypatch.setattr(pc, "_REENTRY_PAGE_BYTES", 31)
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    first = _page(registry, arguments, source["history_start"])
    assert [row["text"] for row in first["rows"]] == [long_text]
    assert first["next_offset"] is not None
    rotate_chat_log_if_needed(tmp_path, max_bytes=1)
    _append(tmp_path, _row(KEY, "new generation, after observation"))
    replay = _page(registry, arguments, source["history_start"])
    assert replay == first
    assert [row["text"] for page in _pages(registry, arguments) for row in page["rows"]] == [
        long_text, "next complete row"]


def test_huge_unicode_row_pages_exact_json_through_the_real_tool_result_cap(tmp_path):
    from ouroboros.loop_tool_execution import _truncate_tool_result

    registry, ctx = _actor(tmp_path)
    cursor = pc._chat_cursor(tmp_path)
    text = "Ω🦉\\\n" * 25_000 + " FINAL WORDS"
    _append(tmp_path, _row(KEY, text))
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    arguments = {**arguments, "presence_reentry_offset": source["history_start"]}

    def consume(args):
        result = registry.execute("get_task_result", args)
        assert len(result) < 24_000
        observed = _truncate_tool_result(result, "get_task_result", args)
        assert observed == result  # the actual loop boundary preserves each bounded tool answer
        return json.loads(observed)["presence_reentry_source"]

    meta = consume(arguments)
    assert meta["range_required"] and meta["row_count"] == 1 and meta["complete_chars"] > 80_000
    assert "rows" not in meta and "text" not in meta
    parts = []
    for start in range(0, meta["complete_chars"], 4000):
        part = consume({**arguments, "source_start_char": start,
                        "source_end_char": min(start + 4000, meta["complete_chars"])})
        assert part["complete_sha256"] == meta["complete_sha256"]
        parts.append(part["text"])
    full = "".join(parts)
    assert hashlib.sha256(full.encode("utf-8")).hexdigest() == meta["complete_sha256"]
    page = json.loads(full)
    assert page["rows"][0]["text"] == text and page["interval_exhausted"]
    assert "read_file" not in {item["function"]["name"] for item in registry.schemas()}


@pytest.mark.parametrize("change", ["truncate", "replace", "rewrite"])
def test_changed_history_is_unavailable_instead_of_an_empty_complete_page(tmp_path, change):
    registry, ctx = _actor(tmp_path)
    cursor = pc._chat_cursor(tmp_path)
    path = _append(tmp_path, _row(KEY, "original source"))
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    if change == "truncate":
        path.write_bytes(b"")
    elif change == "replace":
        path.rename(path.with_name("displaced.jsonl"))
        _append(tmp_path, _row(KEY, "substitute source"))
    else:
        path.write_bytes(path.read_bytes().replace(b"original", b"replaced"))
    page = _page(registry, arguments, source["history_start"])
    assert page["status"] == "unavailable" and page["reason"] == "history_source_changed"
    assert page["rows"] == [] and "interval_exhausted" not in page


def test_pages_disclose_malformed_and_unfinished_rows_and_keep_the_frozen_end(tmp_path, monkeypatch):
    registry, ctx = _actor(tmp_path)
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "whole row"), raw=b'{broken\n[1,2]\n{"text":"unfinished')
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    monkeypatch.setattr(pc, "_REENTRY_PAGE_BYTES", 20)
    # Completing a writer's partial row afterwards does not turn it into an observed complete row.
    _append(tmp_path, raw=b'"}\n')
    pages = _pages(registry, arguments)
    assert [row["text"] for page in pages for row in page["rows"]] == ["whole row"]
    assert [gap["kind"] for page in pages for gap in page["gaps"]] == [
        "jsonl_malformed", "jsonl_non_object", "trailing_row_incomplete"]


def test_offset_cannot_escape_the_bound_interval_or_existing_task_scope(tmp_path):
    from tests.test_presence_own_work import _registry

    registry, ctx = _actor(tmp_path)
    _append(tmp_path, _row(KEY, "before"))
    cursor = pc._chat_cursor(tmp_path)
    _append(tmp_path, _row(KEY, "bound row"))
    arguments = _reader(pc.reentry_note(_binding(tmp_path, cursor), ctx))
    source = json.loads(registry.execute("get_task_result", arguments))["presence_reentry_source"]
    for offset in (True, -1, source["history_start"] - 1, source["history_start"] + 1, source["history_end"] + 1):
        invalid = _page(registry, arguments, offset)
        assert invalid["reason"] == "source_range_invalid" and not invalid["rows"]
    stranger, _ = _registry(tmp_path, binding="b" * 32, task_id="stranger")
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {
        **arguments, "presence_reentry_offset": source["history_start"],
    })
    ended = _page(registry, arguments, source["history_end"])
    assert ended["interval_exhausted"] and ended["next_offset"] is None and ended["rows"] == []
