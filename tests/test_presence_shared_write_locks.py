"""Regression tests for the shared memory writes exercised by Presence turns."""

from __future__ import annotations

import threading
import time

from ouroboros.memory import Memory
from ouroboros.tools.registry import ToolContext


def test_knowledge_topic_and_index_update_share_one_stable_file_lock(tmp_path, monkeypatch):
    from ouroboros.tools import knowledge
    from ouroboros import knowledge as store

    drive = tmp_path / "data"
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive)
    entered_index = threading.Event()
    release_index = threading.Event()
    real_update = store.rebuild_knowledge_index
    calls = 0

    def paused_update(address):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered_index.set()
            assert release_index.wait(2)
        return real_update(address)

    monkeypatch.setattr(store, "rebuild_knowledge_index", paused_update)
    results: list[str] = []
    first = threading.Thread(
        target=lambda: results.append(knowledge._knowledge_write(ctx, "alpha", "one", mode="overwrite")),
    )
    second = threading.Thread(
        target=lambda: results.append(knowledge._knowledge_write(ctx, "beta", "two", mode="overwrite")),
    )

    first.start()
    assert entered_index.wait(2)
    second.start()
    time.sleep(0.05)
    assert not (drive / "memory" / "knowledge" / "beta.md").exists()
    release_index.set()
    first.join(2)
    second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert len(results) == 2
    index = (drive / "memory" / "knowledge" / "index-full.md").read_text(encoding="utf-8")
    assert "**alpha**" in index
    assert "**beta**" in index


def test_scratchpad_source_and_render_stay_under_same_sidecar_lock(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir(parents=True)
    first_memory = Memory(drive)
    second_memory = Memory(drive)
    entered_render = threading.Event()
    release_render = threading.Event()
    real_render = Memory._write_scratchpad_markdown

    def paused_render(self, blocks):
        if len(blocks) == 1 and not entered_render.is_set():
            entered_render.set()
            assert release_render.wait(2)
        return real_render(self, blocks)

    monkeypatch.setattr(Memory, "_write_scratchpad_markdown", paused_render)
    first = threading.Thread(target=first_memory.append_scratchpad_block, args=("older",))
    second = threading.Thread(target=second_memory.append_scratchpad_block, args=("newer",))

    first.start()
    assert entered_render.wait(2)
    second.start()
    time.sleep(0.05)
    assert second.is_alive()
    release_render.set()
    first.join(2)
    second.join(2)

    assert not first.is_alive() and not second.is_alive()
    blocks = first_memory.load_scratchpad_blocks()
    rendered = first_memory.scratchpad_path().read_text(encoding="utf-8")
    assert [block["content"] for block in blocks] == ["older", "newer"]
    assert "older" in rendered and "newer" in rendered
