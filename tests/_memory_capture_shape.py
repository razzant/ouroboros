"""Activated memory installations in the shapes found on the owner's install on 24.10, for counting reads.

Main's view held 241 live rooms; each of its turns read every open row of them, and the memory's
readers looked rows up one at a time. These builders lay the same shapes out in small numbers:
open rows spread over 241 rooms, page writers whose terminal rows a story links, and outgoing rows
written before the lineage epoch whose authors only their task results know.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

MAIN = 1
ROOMS = 241  # Main and the 240 other rooms of the owner's view


def open_rows(root: pathlib.Path, rows: int, *, rooms: int = ROOMS) -> Dict[str, Any]:
    """An activated chronicle, then ``rows`` open rows over ``rooms`` rooms (half of them a task's
    outgoing words, half the owner's); returns Main's direct turn and its view spec."""
    from ouroboros import dialogue_store as ds, memory_view as mv
    from ouroboros.chronicle_store import ChronicleStore

    root.mkdir(parents=True, exist_ok=True)
    assert ChronicleStore(root).ensure_activated()["kind"] == "activation"
    chats = [MAIN, *range(1000, 1000 + rooms - 1)]
    for i in range(rows):
        ds.append(root, "chat", {"chat_id": chats[i % len(chats)], "direction": "in" if i % 2 else "out",
                                 "text": f"message {i}", **({"task_id": f"task{i % 300:04d}"} if i % 2 == 0 else {})})
    task = {"id": "current01", "chat_id": MAIN, "_is_direct_chat": True}
    return {"task": task, "spec": mv.view_spec_for_task(task, root)}


def terminal_links(root: pathlib.Path, pages: int = 40) -> Any:
    """``pages`` pages of Main, each written by its own task, whose later terminal rows (a final
    answer and the host's summary) the story links; returns the chronicle store."""
    from ouroboros import chat_chain as cc, dialogue_store as ds
    from ouroboros.chronicle_store import ChronicleStore

    root.mkdir(parents=True, exist_ok=True)
    store = ChronicleStore(root)
    assert store.ensure_activated()["kind"] == "activation"
    for i in range(pages):
        writer = f"writer{i:03d}"
        row = {"chat_id": MAIN, "direction": "out", "task_id": writer, "text": f"work {i}"}
        ds.append(root, "chat", row)
        published = store.publish_page(room_id=str(MAIN), text=f"page {i}",
                                       covers={"rows": [cc.source_row_id(row)], "stream_span": [i, i]},
                                       author={"kind": "mind", "task_id": writer, "focus": {"role": "root", "task_id": writer}})
        assert published.ok, published
    for i in range(pages):
        writer = f"writer{i:03d}"
        ds.append(root, "chat", {"chat_id": MAIN, "direction": "out", "task_id": writer, "text": "Final",
                                 "terminal_origin": "model_final"})
        ds.append(root, "chat", {"chat_id": MAIN, "direction": "system", "task_id": writer, "text": "Done",
                                 "type": "task_summary", "status": "completed"})
    return store


def legacy_authorship(root: pathlib.Path, tasks: int = 240) -> Dict[str, Any]:
    """Outgoing rows of ``tasks`` root tasks in their own rooms, written before the lineage epoch (the
    old memory covered the first row), so only each task's result says the row is Ouroboros's own."""
    from ouroboros import memory_view as mv
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.utils import jsonl_generation_signature
    from tests._dialogue import import_all, seed
    from tests._store_shared import seed_result

    root.mkdir(parents=True, exist_ok=True)
    rows = [{"chat_id": MAIN, "direction": "in", "text": "covered prompt", "ts": "2026-09-01T00:00:00+00:00"}]
    for i in range(tasks):
        task_id = f"oldtask{i:04d}"
        seed_result(root, task_id, {"status": "completed", "chat_id": MAIN, "root_task_id": task_id, "is_root_task": True})
        rows.append({"chat_id": 1000 + i, "direction": "out", "text": f"old answer {i}", "task_id": task_id,
                     "ts": "2026-09-02T00:00:00+00:00"})
    seed(root, live={"chat": rows})
    memory = root / "memory"
    memory.mkdir()
    (memory / "dialogue_meta.json").write_text(json.dumps({
        "chat_log_signature": jsonl_generation_signature(root / "logs" / "chat.jsonl"), "last_consolidated_offset": 1}))
    (memory / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": "2026-09-02T00:00:00+00:00", "type": "summary", "message_count": 1, "content": "Old prompt",
        "rooms": [{"room_id": str(MAIN), "label": "Main", "message_count": 1, "content": "Old prompt"}]}]))
    import_all(root)
    assert ChronicleStore(root).ensure_activated()["kind"] == "activation"
    task = {"id": "current01", "chat_id": MAIN}
    return {"task": task, "spec": mv.view_spec_for_task(task, root)}
