"""Tests for block-based memory pipeline (scratchpad blocks, dialogue blocks)."""
import json

import pytest

from ouroboros.memory import Memory, _SCRATCHPAD_MAX_BLOCKS


@pytest.fixture
def memory(tmp_path):
    drive = tmp_path / "data"
    drive.mkdir()
    (drive / "memory").mkdir()
    (drive / "logs").mkdir()
    return Memory(drive_root=drive)


class TestScratchpadBlocks:
    def test_load_empty_blocks(self, memory):
        assert memory.load_scratchpad_blocks() == []

    def test_append_block(self, memory):
        block = memory.append_scratchpad_block("test content", source="test")
        assert block["content"] == "test content"
        assert block["source"] == "test"
        blocks = memory.load_scratchpad_blocks()
        assert len(blocks) == 1

    def test_fifo_rotation(self, memory):
        for i in range(_SCRATCHPAD_MAX_BLOCKS + 3):
            memory.append_scratchpad_block(f"block {i}")
        blocks = memory.load_scratchpad_blocks()
        assert len(blocks) == _SCRATCHPAD_MAX_BLOCKS
        assert "block 3" in blocks[0]["content"]

    def test_eviction_journal(self, memory):
        for i in range(_SCRATCHPAD_MAX_BLOCKS + 2):
            memory.append_scratchpad_block(f"block {i}")
        journal = memory.journal_path()
        assert journal.exists()
        lines = [l for l in journal.read_text().strip().split("\n") if l.strip()]
        evictions = [l for l in lines if '"block_evicted"' in l]
        appends = [l for l in lines if '"block_appended"' in l]
        assert len(evictions) == 2
        assert len(appends) == _SCRATCHPAD_MAX_BLOCKS + 2
        entry = json.loads(evictions[0])
        assert entry["type"] == "block_evicted"

    def test_regenerate_scratchpad_md(self, memory):
        memory.append_scratchpad_block("first block")
        memory.append_scratchpad_block("second block")
        md = memory.load_scratchpad()
        assert "second block" in md
        assert "first block" in md
        assert f"{_SCRATCHPAD_MAX_BLOCKS} blocks" in md

    def test_append_block_refuses_retired_legacy_flat_scratchpad(self, memory):
        memory.scratchpad_path().write_text("Legacy scratchpad content here", encoding="utf-8")
        with pytest.raises(RuntimeError, match="LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE"):
            memory.append_scratchpad_block("new block")
        assert memory.load_scratchpad_blocks() == []
        assert "Legacy scratchpad" in memory.scratchpad_path().read_text(encoding="utf-8")
        assert "legacy_scratchpad_requires_manual_upgrade" in memory.journal_path().read_text(encoding="utf-8")

    def test_append_block_from_default_scratchpad(self, memory):
        memory.ensure_files()
        memory.append_scratchpad_block("new block")
        blocks = memory.load_scratchpad_blocks()
        assert len(blocks) == 1
        assert blocks[0]["source"] == "task"


class TestDialogueBlocks:
    def test_load_empty(self, memory):
        assert memory.load_dialogue_blocks() == []

    def test_load_blocks(self, memory):
        blocks_path = memory.drive_root / "memory" / "dialogue_blocks.json"
        blocks = [
            {"type": "summary", "content": "First block", "range": "2026-01-01"},
            {"type": "era", "content": "Old era", "range": "2025-12-01 to 2025-12-31"},
        ]
        blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
        loaded = memory.load_dialogue_blocks()
        assert len(loaded) == 2

    def test_corrupt_blocks_file(self, memory):
        blocks_path = memory.drive_root / "memory" / "dialogue_blocks.json"
        blocks_path.write_text("NOT VALID JSON {{{", encoding="utf-8")
        assert memory.load_dialogue_blocks() == []


class TestCacheHitRate:
    def test_cache_hit_rate_computation(self, memory):
        from ouroboros.context import _compute_cache_hit_rate
        import types

        events_path = memory.logs_path("events.jsonl")
        events_path.parent.mkdir(parents=True, exist_ok=True)

        entries = []
        for i in range(10):
            entries.append(json.dumps({
                "type": "llm_round",
                "prompt_tokens": 1000,
                "cached_tokens": 600,
                "cache_hit_rate": 0.6,
            }))
        events_path.write_text("\n".join(entries), encoding="utf-8")

        env = types.SimpleNamespace(
            drive_path=lambda p: memory.drive_root / p,
        )
        rate = _compute_cache_hit_rate(env)
        assert rate is not None
        assert abs(rate - 0.6) < 0.01

    def test_cache_hit_rate_insufficient_data(self, memory):
        from ouroboros.context import _compute_cache_hit_rate
        import types

        env = types.SimpleNamespace(
            drive_path=lambda p: memory.drive_root / p,
        )
        rate = _compute_cache_hit_rate(env)
        assert rate is None
