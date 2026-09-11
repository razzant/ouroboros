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


# ---------------------------------------------------------------------------
# ibl-28109914789e — defer on transient empty / non-JSON model reply
# ---------------------------------------------------------------------------

def _seed_scratchpad_blocks(tmp_path, count: int = 3) -> tuple[list[dict[str, object]], object]:
    """Drop three pre-existing scratchpad blocks onto disk for consolidation.

    Returns ``(blocks, memory)`` — the blocks already live on disk via
    ``Memory.mutate_scratchpad_blocks`` so the consolidator's re-read-under-lock
    sees them too.

    Each block's content is padded past ``SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS``
    (30,000) so the consolidator actually enters its LLM call path; a 3-block
    "block N" seed would otherwise early-return None before our deferral code
    runs.
    """
    from ouroboros.memory import Memory

    memory = Memory(drive_root=tmp_path)
    pad = "x" * 12_000  # 3 * 12,000 = 36,000 > 30,000 threshold
    blocks = [
        {
            "ts": f"2026-09-01T0{i}:00:00Z",
            "source": "test",
            "content": f"block {i} " + pad,
        }
        for i in range(count)
    ]
    memory.mutate_scratchpad_blocks(lambda _live: list(blocks))
    return blocks, memory


def _memory_with_blocks(tmp_path):
    from ouroboros.memory import Memory

    return Memory(drive_root=tmp_path)


def test_consolidate_scratchpad_blocks_defers_on_empty_response(tmp_path, caplog):
    """ibl-28109914789e — model returning ``""`` (e.g. 429-fallback empty body)
    used to error the whole pass and return None, wedging the consolidator.

    After the fix: empty response is a recoverable transient — the pass returns
    the usage dict, logs a warning containing "empty", and DOES NOT write
    knowledge or mutate scratchpad blocks."""
    import logging

    seeded, memory = _seed_scratchpad_blocks(tmp_path)
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = tmp_path / "memory" / "scratchpad_blocks.json"
    before = json.loads(blocks_path.read_text())

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        {"content": ""},
        {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10, "cost": 0.0},
    )

    from ouroboros.consolidator import _consolidate_scratchpad_blocks

    caplog.set_level(logging.WARNING, logger="ouroboros.consolidator")
    result = _consolidate_scratchpad_blocks(
        memory=memory,
        blocks=seeded,
        knowledge_dir=knowledge_dir,
        llm_client=mock_llm,
        identity_text="",
    )

    # Clean deferral: usage returned, NOT None.
    assert result is not None
    assert result["prompt_tokens"] == 10
    # No knowledge files were created.
    assert not any(knowledge_dir.glob("*.md"))
    # Scratchpad blocks on disk are unchanged (no merge ran).
    after = json.loads(blocks_path.read_text())
    assert after == before
    # And the warning was loud enough to name "empty".
    assert any("empty" in rec.message for rec in caplog.records)
    # And NOT the loud ERROR path the old code took.
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_consolidate_scratchpad_blocks_defers_on_non_json_prose(tmp_path, caplog):
    """ibl-28109914789e — non-JSON prose reply must defer cleanly too."""
    import logging

    seeded, memory = _seed_scratchpad_blocks(tmp_path)
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = tmp_path / "memory" / "scratchpad_blocks.json"
    before = json.loads(blocks_path.read_text())

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        {"content": "sorry I can't help with that right now"},
        {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17, "cost": 0.0001},
    )

    from ouroboros.consolidator import _consolidate_scratchpad_blocks

    caplog.set_level(logging.WARNING, logger="ouroboros.consolidator")
    result = _consolidate_scratchpad_blocks(
        memory=memory,
        blocks=seeded,
        knowledge_dir=knowledge_dir,
        llm_client=mock_llm,
        identity_text="",
    )

    assert result is not None
    assert result["prompt_tokens"] == 12
    # No knowledge writes, no scratchpad mutation.
    assert not any(knowledge_dir.glob("*.md"))
    after = json.loads(blocks_path.read_text())
    assert after == before
    # Warning names the deferred reason (the first 120 chars of raw).
    assert any("non-JSON" in rec.message and "sorry I can" in rec.message
               for rec in caplog.records)
    # Never an ERROR.
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_consolidate_scratchpad_blocks_recovers_embedded_json(tmp_path, caplog):
    """A fenced ```` ```json\\n{...}\\n``` ```` payload is the existing happy path.
    Also: a prose reply that wraps JSON inside braces still parses via the
    bounded embedded-recovery path."""
    seeded, memory = _seed_scratchpad_blocks(tmp_path)
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = tmp_path / "memory" / "scratchpad_blocks.json"

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        {"content": (
            "Here you go:\n\n"
            "```json\n"
            "{\"compressed_block\": \"c\", \"knowledge_entries\": []}\n"
            "```\n"
        )},
        {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12, "cost": 0.0001},
    )

    from ouroboros.consolidator import _consolidate_scratchpad_blocks

    result = _consolidate_scratchpad_blocks(
        memory=memory,
        blocks=seeded,
        knowledge_dir=knowledge_dir,
        llm_client=mock_llm,
        identity_text="",
    )

    assert result is not None
    # Compression ran: scratchpad now holds the new compressed block.
    blocks = json.loads(blocks_path.read_text())
    assert any(b.get("source") == "consolidation" for b in blocks)


def test_consolidate_scratchpad_blocks_valid_bare_json(tmp_path, caplog):
    """A genuinely valid bare-JSON payload is the existing happy path —
    parses and consolidates normally."""
    seeded, memory = _seed_scratchpad_blocks(tmp_path)
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = tmp_path / "memory" / "scratchpad_blocks.json"

    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        {"content": json.dumps({"compressed_block": "x", "knowledge_entries": []})},
        {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0},
    )

    from ouroboros.consolidator import _consolidate_scratchpad_blocks

    result = _consolidate_scratchpad_blocks(
        memory=memory,
        blocks=seeded,
        knowledge_dir=knowledge_dir,
        llm_client=mock_llm,
        identity_text="",
    )

    assert result is not None
    blocks = json.loads(blocks_path.read_text())
    assert any(b.get("source") == "consolidation" for b in blocks)


def test_try_extract_embedded_json_balanced_and_skips_strings(tmp_path):
    """Helper sanity: braces inside JSON strings do NOT count toward depth; the
    outermost matching brace is picked; a top-level non-dict returns None."""
    from ouroboros.consolidator import _try_extract_embedded_json

    payload = (
        'Sure! Here: {"a": "}", "b": 1, "c": {"d": 2}}\n\nThanks.'
    )
    parsed = _try_extract_embedded_json(payload)
    assert parsed == {"a": "}", "b": 1, "c": {"d": 2}}

    # Top-level list, not dict — defer.
    assert _try_extract_embedded_json("[1, 2, 3]") is None
    # Empty / no braces — defer.
    assert _try_extract_embedded_json("") is None
    assert _try_extract_embedded_json("no json here at all") is None
    # Unbalanced — defer.
    assert _try_extract_embedded_json("{ \"a\": 1 ") is None
