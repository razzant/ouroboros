"""Tests for block-based memory pipeline (scratchpad blocks, dialogue blocks)."""
import json

import pytest

from ouroboros.memory import Memory, scratchpad_block_sha256


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

    def test_no_rotation_past_the_former_ten_block_cap(self, memory):
        # #918 (owner, 29 Oct 2026): the former FIFO evicted "block 0".."block 2" here.
        for i in range(13):
            memory.append_scratchpad_block(f"block {i}")
        blocks = memory.load_scratchpad_blocks()
        assert [block["content"] for block in blocks] == [f"block {i}" for i in range(13)]

    def test_append_journal_records_appends_and_no_eviction(self, memory):
        for i in range(12):
            memory.append_scratchpad_block(f"block {i}")
        journal = memory.journal_path()
        assert journal.exists()
        rows = [json.loads(line) for line in journal.read_text().strip().split("\n") if line.strip()]
        assert [row["type"] for row in rows] == ["block_appended"] * 12
        assert rows[-1]["content_len"] == sum(len(f"block {i}") for i in range(12))

    def test_regenerate_scratchpad_md(self, memory):
        memory.append_scratchpad_block("first block")
        memory.append_scratchpad_block("second block")
        md = memory.load_scratchpad()
        assert "second block" in md
        assert "first block" in md
        assert "2 blocks" in md and "/10" not in md
        for block in memory.load_scratchpad_blocks():
            assert f"sha256 {scratchpad_block_sha256(block)}" in md

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
