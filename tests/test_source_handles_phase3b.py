"""Production-shaped source-handle regressions for continuity Phase 3B."""

from __future__ import annotations

import json

from ouroboros.memory import Memory, scratchpad_block_sha256
from ouroboros.tools.core import _read_file
from ouroboros.tools.core_file_tools import delivered_source_prefix
from ouroboros.tools.registry import ToolContext


def _tool_ctx(tmp_path, *, task_id: str = "source-handles") -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task_id)


def _read_source(ctx: ToolContext, ref: dict, *, start_char: int = 0) -> str:
    read = ref["read"]
    assert read["tool"] == "read_file"
    args = dict(read["arguments"])
    args["start_char"] = start_char
    display = _read_file(ctx, **args)
    return delivered_source_prefix(ctx.last_read_view, display, len(display)) if ctx.last_read_view else display


def test_eleven_blocks_stay_whole_and_a_fold_names_its_exact_journaled_originals(tmp_path):
    """#918: the former FIFO evicted ``scratch-source-0`` here; now all eleven stay, and a fold's
    exact originals are what the current scratchpad's journal reader returns."""
    memory = Memory(tmp_path)
    contents = [f"scratch-source-{index}" for index in range(11)]
    for content in contents:
        memory.append_scratchpad_block(content, source="phase3b-test")
    assert [block["content"] for block in memory.load_scratchpad_blocks()] == contents

    first = memory.load_scratchpad_blocks()[0]
    memory.fold_scratchpad_blocks("folded scratch-source-0", [scratchpad_block_sha256(first)])

    current = memory.load_scratchpad()
    assert f"Exact replaced blocks: {memory.scratchpad_journal_reader()};" in current
    journal = _read_source(_tool_ctx(tmp_path), {"read": {"tool": "read_file", "arguments": {
        "root": "runtime_data", "path": "memory/scratchpad_journal.jsonl"}}})
    rows = [json.loads(line) for line in journal.splitlines() if line.startswith("{")]
    assert [row["source_blocks"] for row in rows if row.get("type") == "fold_sources_preserved"] == [[first]]
    assert not [row for row in rows if row.get("type") == "block_evicted"]


# ── AP3: a pageable tool result persists its exact source like any other ──────
