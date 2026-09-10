"""Tests for ouroboros.consolidator (block-wise system)."""
import json
import pathlib
import pytest
from unittest.mock import MagicMock

from ouroboros.consolidator import (
    should_consolidate,
    consolidate,
    _load_meta,
    atomic_write_json as _save_meta,
    _count_lines,
    _pending_bytes,
    _format_entries_for_block,
    _load_blocks,
    _write_locked_json as _save_blocks,
    BLOCK_SIZE,
)


@pytest.fixture
def tmp_paths(tmp_path):
    chat_path = tmp_path / "chat.jsonl"
    blocks_path = tmp_path / "dialogue_blocks.json"
    meta_path = tmp_path / "dialogue_meta.json"
    return chat_path, blocks_path, meta_path


def _write_chat_entries(path, count):
    """Write count fake chat entries."""
    with path.open("w") as f:
        for i in range(count):
            direction = "in" if i % 2 == 0 else "out"
            entry = {
                "ts": f"2026-02-25T{10 + i // 60:02d}:{i % 60:02d}:00Z",
                "direction": direction,
                "text": f"Message {i}",
            }
            f.write(json.dumps(entry) + "\n")


def test_should_consolidate_no_chat(tmp_paths):
    _, _, meta_path = tmp_paths
    chat_path = pathlib.Path("/nonexistent/chat.jsonl")
    assert should_consolidate(meta_path, chat_path) is False


def test_should_consolidate_not_enough_messages(tmp_paths):
    chat_path, _, meta_path = tmp_paths
    _write_chat_entries(chat_path, 5)
    assert should_consolidate(meta_path, chat_path) is False


def test_should_consolidate_enough_messages(tmp_paths):
    chat_path, _, meta_path = tmp_paths
    _write_chat_entries(chat_path, BLOCK_SIZE + 5)
    assert should_consolidate(meta_path, chat_path) is True


def test_should_consolidate_respects_offset(tmp_paths):
    chat_path, _, meta_path = tmp_paths
    _write_chat_entries(chat_path, BLOCK_SIZE + 5)
    _save_meta(meta_path, {"last_consolidated_offset": BLOCK_SIZE + 2})
    assert should_consolidate(meta_path, chat_path) is False


def test_pending_bytes_ignores_segments_covered_by_cursor(tmp_path):
    """ibl-f60344038572: _pending_bytes must walk segments from the chain
    cursor and not double-count already-consumed prefix bytes. A large archive
    whose cursor already covers must NOT contribute its st_size.
    """
    seg0 = tmp_path / "live.jsonl"
    seg1 = tmp_path / "archive.jsonl"
    seg0.write_text("\n".join('{"k":"v"}' for _ in range(50)) + "\n")  # ~400 bytes, 50 lines
    seg1.write_bytes(b"x" * 100_000)  # 100KB archive, single line

    seg0_size = seg0.stat().st_size
    seg0_lines = _count_lines(seg0)
    seg1_lines = _count_lines(seg1)  # 1 line

    # Case 1: cursor at start (last_offset=0) → all of seg0 + all of seg1 pending
    assert _pending_bytes([seg0, seg1], 0) == seg0_size + 100_000

    # Case 2: cursor at end of seg0 → seg0 contributes 0, seg1 fully pending
    assert _pending_bytes([seg0, seg1], seg0_lines) == 100_000

    # Case 3: cursor past all segments → 0 pending (degenerate; nothing left)
    assert _pending_bytes([seg0, seg1], seg0_lines + seg1_lines) == 0

    # Case 4: cursor mid-seg0 → proportional share of seg0 + all of seg1
    half_offset = seg0_lines // 2
    expected = int(seg0_size * 0.5) + 100_000
    assert _pending_bytes([seg0, seg1], half_offset) == expected

    # Case 5: empty / missing segments → 0 (no crash)
    assert _pending_bytes([], 0) == 0
    assert _pending_bytes([tmp_path / "missing.jsonl"], 0) == 0


def test_should_consolidate_archived_latch_off(tmp_paths):
    """ibl-f60344038572 integration: even with archived st_size > PENDING_BYTES_TRIGGER
    alone, pending_lines < BLOCK_SIZE and cursor-past-archive must return False.
    """
    chat_path, _, meta_path = tmp_paths
    # Write a few live messages — well under BLOCK_SIZE lines
    _write_chat_entries(chat_path, 5)
    # Save an offset that pretends we've already consolidated past the live region
    _save_meta(meta_path, {"last_consolidated_offset": 5})
    # Five lines, cursor at line 5 → pending_lines=0, far below BLOCK_SIZE
    assert should_consolidate(meta_path, chat_path) is False


def test_consolidate_creates_block(tmp_paths):
    chat_path, blocks_path, meta_path = tmp_paths
    _write_chat_entries(chat_path, BLOCK_SIZE + 5)

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        {"content": "### Block: 2026-02-25 10:00 - 11:40\n\nSummary of events."},
        {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001},
    )

    usage = consolidate(chat_path, blocks_path, meta_path, mock_llm)

    assert usage is not None
    assert usage["cost"] == 0.001
    assert blocks_path.exists()
    blocks = json.loads(blocks_path.read_text())
    assert len(blocks) == 1
    assert blocks[0]["type"] == "summary"
    assert "Summary of events" in blocks[0]["content"]

    meta = _load_meta(meta_path)
    assert meta["last_consolidated_offset"] == BLOCK_SIZE


def test_consolidate_not_enough_messages(tmp_paths):
    chat_path, blocks_path, meta_path = tmp_paths
    _write_chat_entries(chat_path, 5)

    mock_llm = MagicMock()
    result = consolidate(chat_path, blocks_path, meta_path, mock_llm)
    assert result is None
    mock_llm.chat.assert_not_called()


def test_load_save_meta(tmp_paths):
    _, _, meta_path = tmp_paths
    assert _load_meta(meta_path) == {}

    _save_meta(meta_path, {"last_consolidated_offset": 42})
    meta = _load_meta(meta_path)
    assert meta["last_consolidated_offset"] == 42


def test_count_lines(tmp_paths):
    chat_path = tmp_paths[0]
    _write_chat_entries(chat_path, 15)
    assert _count_lines(chat_path) == 15


def test_read_chat_entries_includes_project_rows_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the one identity's consolidated dialogue
    (dialogue_blocks.json) is its WHOLE conversation — main AND project threads —
    because Ouroboros is one awareness/biography (BIBLE P1). Only A2A virtual
    transport is excluded from the consolidation source."""
    from ouroboros.consolidator import _read_chat_entries
    from ouroboros.projects_registry import create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    chat_path = logs / "chat.jsonl"
    rows = [
        {"chat_id": 1, "direction": "in", "text": "main-1"},
        {"chat_id": project_chat, "direction": "in", "text": "project-visible"},
        {"chat_id": 1, "direction": "out", "text": "main-2"},
        {"chat_id": -1001, "direction": "out", "text": "a2a-noise"},
    ]
    chat_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    texts = [e.get("text") for e in _read_chat_entries(chat_path)]
    assert "main-1" in texts and "main-2" in texts
    assert "project-visible" in texts  # full awareness: project chat IS part of identity memory
    assert "a2a-noise" not in texts    # only A2A virtual transport is excluded


def test_should_consolidate_handles_log_rotation(tmp_paths):
    chat_path, _, meta_path = tmp_paths
    _write_chat_entries(chat_path, BLOCK_SIZE + 5)
    _save_meta(meta_path, {"last_consolidated_offset": 9999})
    assert should_consolidate(meta_path, chat_path) is True


def test_format_entries_for_block():
    entries = [
        {"ts": "2026-02-25T10:00:00Z", "direction": "in", "text": "Hello"},
        {"ts": "2026-02-25T10:01:00Z", "direction": "out", "text": "Hi there"},
    ]
    formatted = _format_entries_for_block(entries)
    assert "User: Hello" in formatted
    assert "Ouroboros: Hi there" in formatted


def test_load_save_blocks(tmp_path):
    blocks_path = tmp_path / "blocks.json"
    blocks = [{"ts": "2026-01-01", "type": "summary", "content": "test"}]
    _save_blocks(blocks_path, blocks)
    loaded = _load_blocks(blocks_path)
    assert len(loaded) == 1
    assert loaded[0]["content"] == "test"


def test_dialogue_summary_markdown_no_longer_auto_migrates(tmp_path):
    summary_path = tmp_path / "dialogue_summary.md"
    blocks_path = tmp_path / "dialogue_blocks.json"

    summary_path.write_text(
        "### Episode: 2026-02-20 10:00 – 11:00\n\nFirst episode.\n\n"
        "### Era: 2026-01-01 to 2026-01-31\n\nOld era summary.\n",
        encoding="utf-8",
    )

    assert summary_path.exists()
    assert not blocks_path.exists()


def test_existing_dialogue_blocks_remain_authoritative(tmp_path):
    summary_path = tmp_path / "dialogue_summary.md"
    blocks_path = tmp_path / "dialogue_blocks.json"

    summary_path.write_text("### Episode: test\nContent.", encoding="utf-8")
    blocks_path.write_text("[]", encoding="utf-8")

    assert json.loads(blocks_path.read_text()) == []


def test_strip_think_blocks_basic():
    """The most common case: a single <think>...</think> block at the start."""
    from ouroboros.consolidator import _strip_think_blocks

    raw = "<think>\nplan: foo\n</think>\n\n### Block: 2026-09-07\nactual content"
    assert _strip_think_blocks(raw) == "### Block: 2026-09-07\nactual content"


def test_strip_think_blocks_no_think():
    """No think block → raw returned unchanged."""
    from ouroboros.consolidator import _strip_think_blocks

    raw = "### Block: 2026-09-07\nactual content"
    assert _strip_think_blocks(raw) == raw


def test_strip_think_blocks_multiple():
    """Multiple disjoint think blocks are all stripped."""
    from ouroboros.consolidator import _strip_think_blocks

    raw = (
        "<think>plan 1</think>summary intro\n"
        "<think>plan 2</think>body content\n"
        "trailing text"
    )
    out = _strip_think_blocks(raw)
    assert "plan 1" not in out
    assert "plan 2" not in out
    assert "summary intro" in out
    assert "body content" in out
    assert "trailing text" in out


def test_strip_think_blocks_unclosed_fails_safe():
    """Unclosed ``<think>`` (no closing tag) must NOT eat the whole reply —
    the conservative regex is paired-aware. Without a closing tag the think
    block is NOT stripped and the whole raw text passes through."""
    from ouroboros.consolidator import _strip_think_blocks

    raw = "<think>never closed\nactual content here"
    out = _strip_think_blocks(raw)
    assert out == raw  # unclosed → pass-through, not silently consumed


def test_strip_think_blocks_multiline():
    """Multiline think content with DOTALL semantics is removed as a unit."""
    from ouroboros.consolidator import _strip_think_blocks

    raw = (
        "<think>\n"
        "first line of plan\n"
        "second line\n"
        "third line\n"
        "</think>\n"
        "### Block\nactual"
    )
    out = _strip_think_blocks(raw)
    assert out == "### Block\nactual"


def test_truncate_with_marker_under_cap():
    """Content below the cap returns unchanged."""
    from ouroboros.consolidator import _truncate_with_marker

    text = "short content " * 10  # ~150 chars
    out, truncated = _truncate_with_marker(text, max_chars=1000, span_label="test")
    assert truncated is False
    assert out == text


def test_truncate_with_marker_over_cap():
    """Content over the cap is truncated with an honest marker naming the cut span."""
    from ouroboros.consolidator import _truncate_with_marker

    text = "x" * 5_000
    out, truncated = _truncate_with_marker(text, max_chars=2_000, span_label="block 2026-09-07")
    assert truncated is True
    assert out.startswith("x" * 2_000)
    assert "truncated" in out.lower()
    assert "block 2026-09-07" in out


def test_truncate_with_marker_exact_cap():
    """Content exactly at the cap does NOT trigger truncation."""
    from ouroboros.consolidator import _truncate_with_marker

    text = "x" * 1_000
    out, truncated = _truncate_with_marker(text, max_chars=1_000, span_label="test")
    assert truncated is False
    assert out == text


def test_call_consolidation_llm_strips_think_and_truncates(tmp_paths):
    """End-to-end: _call_consolidation_llm returns content with think stripped
    AND over-cap content truncated."""
    from ouroboros.consolidator import _call_consolidation_llm

    mock_llm = MagicMock()
    overlong_body = "x" * 30_000
    mock_llm.chat.return_value = (
        {
            "content": (
                "<think>\nplan: blah\nblah\n</think>\n\n"
                f"### Block: 2026-09-07\n{overlong_body}"
            )
        },
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.001},
    )

    out, usage = _call_consolidation_llm(
        mock_llm, "any prompt", "test", max_chars=24_000, span_label="block 2026-09-07"
    )

    assert "plan: blah" not in out
    assert out.startswith("### Block: 2026-09-07")
    assert "truncated" in out.lower()
    # Content is bounded: marker + body < 24_000 + small marker overhead.
    assert len(out) <= 24_000 + 500
    assert usage["cost"] == 0.001


def test_call_consolidation_llm_under_cap_unchanged():
    """Under-cap content passes through clean — no truncation marker, no think
    bleed."""
    from ouroboros.consolidator import _call_consolidation_llm

    mock_llm = MagicMock()
    body = "### Block: 2026-09-07\nclean summary body, no think, well under cap."
    mock_llm.chat.return_value = (
        {"content": body},
        {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15, "cost": 0.0},
    )

    out, _usage = _call_consolidation_llm(
        mock_llm, "any prompt", "test", max_chars=24_000, span_label="block 2026-09-07"
    )

    assert out == body


def test_should_consolidate_byte_trigger(tmp_paths, monkeypatch):
    """Byte-aware secondary trigger: pending volume below BLOCK_SIZE but raw
    bytes >= PENDING_BYTES_TRIGGER fires the consolidator (catches pathological
    long-message runs)."""
    from ouroboros import consolidator as cons_mod
    monkeypatch.setattr(cons_mod, "PENDING_BYTES_TRIGGER", 5_000)

    chat_path, _, meta_path = tmp_paths
    # 50 messages (under BLOCK_SIZE=100) but each carries a huge payload → ~12.5KB
    _write_big_chat_entries(chat_path, count=50, payload_size=250)
    assert should_consolidate(meta_path, chat_path) is True


def test_should_consolidate_no_byte_trigger_when_small(tmp_paths):
    """The byte trigger does NOT fire when raw bytes are small even with
    pending lines below BLOCK_SIZE — only one of (line cap OR byte cap) needs
    to hold."""

    chat_path, _, meta_path = tmp_paths
    _write_big_chat_entries(chat_path, count=10, payload_size=10)
    assert should_consolidate(meta_path, chat_path) is False


def _write_big_chat_entries(path, *, count: int, payload_size: int):
    """Write count chat entries each carrying payload_size bytes of text."""
    import json as _json

    payload = "x" * payload_size
    with path.open("w") as f:
        for i in range(count):
            entry = {
                "ts": f"2026-09-07T{10 + i // 60:02d}:{i % 60:02d}:00Z",
                "direction": "in" if i % 2 == 0 else "out",
                "text": payload,
            }
            f.write(_json.dumps(entry) + "\n")

