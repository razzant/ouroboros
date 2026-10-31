"""The history, activity, breadth and rows the cost contract's turn law (B10) lays on a booted root.

tests/_cost_driver.py lays them after the archive import, untagged, between the turns it measures
(tests/test_cost_contract.py, B10). Every chat row goes in through ONE store transaction that runs the
statements ``dialogue_store.append`` runs for each (tests/test_cost_meter.py holds the two to the same
stored row) and every page in ONE publication (a part folds against its room's head, one publication per
room), so a k=30 root takes seconds, not minutes. After the history the install is brought to its steady
state (``steady_state``): what its maintenance child and its boot leave before an owner's turn.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List

SEALED_ROWS = 12  # per unit of k: rows of a room after the frontier, all sealed by one page
OPEN_ROWS = 3  # per breadth room: rows no page seals
BREADTH = 40  # live rooms the breadth step adds at k=1, with a quarter as many running roots
MORE_ROWS = 650  # open rows the rows step adds over the same rooms (B9's 650 -> 1,300, in the real turn)
ARCHIVED = 40  # archived knowledge notes and old rescue snapshots the archive step adds at k=1
RESCUE_AGE_SEC = 3 * 86_400.0  # an old restart's rescue snapshot: older than the health block's two hours
CATALOGUE_WINDOW = 1_000_000  # tokens: more than any view measured here fills, so the fit keeps Max as before


def _author(task_id: str) -> Dict[str, Any]:
    return {"kind": "mind", "task_id": task_id, "focus": {"role": "root", "task_id": task_id}}


def _rows(chat_id: int, count: int, tag: str) -> List[Dict[str, Any]]:
    from ouroboros.utils import utc_now_iso

    now = utc_now_iso()
    return [{"chat_id": chat_id, "direction": "out" if i % 2 else "in", "text": f"{tag} {chat_id}-{i}", "ts": now,
             **({"task_id": f"w{chat_id}"} if i % 2 else {})} for i in range(count)]


def append_rows(root: pathlib.Path, rows: List[Dict[str, Any]]) -> List[str]:
    """Chat rows in ONE store transaction, through the statements ``dialogue_store.append`` runs for each
    (tests/test_cost_meter.py holds the two to the same stored row); their source row ids."""
    from ouroboros import chat_chain, dialogue_store as ds

    with ds.transaction(root, write=True) as conn:
        for row in rows:
            body, columns = ds.encode(row), ds.facts(row)
            length, counted = len(body.encode("utf-8")) + 1, ds.stream_row("chat", columns)
            marks = conn.execute(
                "SELECT next_seq, live, next_ordinal, next_pos FROM streams WHERE stream='chat'").fetchone()
            seq = int(marks["next_seq"])
            ds.insert_row(conn, "chat", seq, row, body, length=length, segment=int(marks["live"]),
                          ordinal=int(marks["next_ordinal"]), pos=int(marks["next_pos"]) if counted else None)
            conn.execute("UPDATE streams SET next_seq=?, next_ordinal=next_ordinal+1, next_pos=next_pos+? "
                         "WHERE stream='chat'", (seq + length, int(counted)))
    return [chat_chain.source_row_id(row) for row in rows]


def _page(room: Any, ids: List[str], text: str) -> Dict[str, Any]:
    room = str(room)
    return {"kind": "page", "room_id": room, "text": text, "author": _author(f"w{room}"), "metadata": {},
            "covers": {"room_id": room, "rows": ids, "stream_span": [0, len(ids) - 1]}}


def _old_memory(root: pathlib.Path, k: int) -> int:
    """The old memory the activation imports: one block over the stream's oldest rows (the requested room's,
    constant) whose retelling has a section for each of k rooms that had no row in it, so a part folds each.
    Its message count meets the old cursor, so the range is exact and the frontier lies after those rows."""
    from ouroboros import chat_chain

    oldest = 0
    for _address, row, _pos in chat_chain.iter_rows(root):
        if not str(row.get("ts") or "").startswith("2024-"):
            break
        oldest += 1
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "dialogue_meta.json").write_text(json.dumps({
        "chat_log_signature": {"first_line_sha256": "dialogue:chat"}, "last_consolidated_offset": oldest}),
        encoding="utf-8")
    (memory / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": "2024-01-02T00:00:00+00:00", "type": "summary", "message_count": oldest, "content": "The first days.",
        "rooms": [{"room_id": str(2000 + unit), "label": f"Room {2000 + unit}", "message_count": 0,
                   "content": f"What room {2000 + unit} settled before the chronicle began."} for unit in range(k)]}]),
        encoding="utf-8")
    return oldest


def archive_history(root: pathlib.Path, n: int) -> Dict[str, Any]:
    """n knowledge notes the mind archived and n old restarts' rescue snapshots, as old as their dates: their files
    and the rescue folder carry an old time, the archive folder the time the last log rotation left it (the rescue
    folder was there)."""
    shelf, archive = root / "memory" / "knowledge", root / "archive"
    shelf.mkdir(parents=True, exist_ok=True)
    rotated, old = archive.stat(), time.time() - RESCUE_AGE_SEC
    for unit in range(n):
        note = shelf / f"campaign-{unit:03d}.md"
        note.write_text(f"---\ntitle: Campaign {unit}\nsummary: what campaign {unit} learned\narchive:\n"
                        f"  at: '2025-01-02T00:00:00+00:00'\n  reason: superseded by the current notes\n---\n"
                        f"# Campaign {unit}\n\n" + "A finding of that campaign, kept for its history. " * 40,
                        encoding="utf-8")
        snapshot = archive / "rescue" / f"20250101T{unit:06d}"
        snapshot.mkdir(parents=True)
        (snapshot / "rescue_meta.json").write_text(json.dumps({"reason": "restart", "files": 1}), encoding="utf-8")
        (snapshot / "changes.diff").write_text("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
        for path in (note, snapshot / "rescue_meta.json", snapshot / "changes.diff", snapshot):
            os.utime(path, (old, old))
    os.utime(archive / "rescue", (old, old))
    os.utime(archive, ns=(rotated.st_atime_ns, rotated.st_mtime_ns))
    return {"notes": n, "snapshots": n, "rows": 0}


def _rotated_logs(root: pathlib.Path) -> List[str]:
    """A live install's logs rotate (``supervisor.state.rotate_jsonl_log_if_needed``, at 800 KB), and a turn reads
    each live file's tail plus at most its three newest archives. Rotated here as the server rotates them, the live
    files the turns read hold the same rows at both sizes; what the build and the boot wrote, k-sized, is archived."""
    from ouroboros.utils import append_jsonl, utc_now_iso
    from supervisor.state import rotate_jsonl_log_if_needed

    names = ["events", "tools", "supervisor"]
    for name in names:  # the append that rotates a log lands in its new live file, which a reader then keys on
        rotate_jsonl_log_if_needed(root, f"{name}.jsonl", name, max_bytes=0)
        append_jsonl(root / "logs" / f"{name}.jsonl", {"ts": utc_now_iso(), "type": "llm_round", "task_id": "cur-on"})
    return names


def _consume_closed_outputs(root: pathlib.Path, k: int) -> int:
    """The scale root's closed runs are copies of a run whose output was never read, so each would stand as a
    current obligation; history's closed runs had theirs consumed."""
    from ouroboros import store
    from tests._scale_root import CLOSED_RUNS_PER_K, CUSTODY_PER_K

    ids = [f"run-hist-{unit}-{index}" for unit in range(k) for index in range(CUSTODY_PER_K)]
    ids += [f"run-closed-{unit}-{index}" for unit in range(k) for index in range(CLOSED_RUNS_PER_K)]
    with store.open_store(root, write=True) as txn:
        for run_id in ids:
            txn.put_custody(run_id, {**txn.custody_row(run_id), "output_consumed": True}, state="closed")
    return len(ids)


def turn_history(root: pathlib.Path, k: int) -> Dict[str, Any]:
    """History a turn must not reread, every row in ONE transaction and every page in ONE publication: the old
    memory imported at activation, its k sections folded by parts; the root's archived dialogue and k rooms whose
    rows came after the frontier, sealed by a page per day in every room a row is in (``rooms_of_row``: Main holds
    the unbound rows of every chat outside Projects, the chat its own); the closed runs' outputs consumed; the logs
    rotated."""
    from ouroboros import chat_chain, memory_inventory
    from ouroboros.chronicle_store import ChronicleStore

    oldest = _old_memory(root, k)
    store = ChronicleStore(root)
    activation = store.ensure_activated(wait=True)
    fresh = {2000 + unit: _rows(2000 + unit, SEALED_ROWS, "sealed") for unit in range(k)}
    sealed = set(append_rows(root, [row for rows in fresh.values() for row in rows]))
    facts, groups = memory_inventory.membership_facts(root), {}
    for _address, row, _pos in chat_chain.iter_rows(root):
        row_id, ts = chat_chain.source_row_id(row), str(row.get("ts") or "")
        if ts.startswith("2025-") or row_id in sealed:
            for room in memory_inventory.rooms_of_row(memory_inventory.row_meta(row), facts):
                groups.setdefault((room, ts[:10]), []).append(row_id)
    records = [_page(room, ids, f"page of room {room} on {day}") for (room, day), ids in sorted(groups.items())]
    published = store.publish(records)
    if not published.ok:
        raise RuntimeError(f"sealing pages: {published}")
    for chat_id in fresh:  # a part folds the room's retelling: one publication per room, against its head
        folded = store.publish_part(room_id=chat_id, text=f"the old story of room {chat_id}",
                                    member_ids=[f"legacy-b00-r{chat_id}"], author=_author(f"w{chat_id}"),
                                    expected_sequence=store.room_head(chat_id))
        if not folded.ok:
            raise RuntimeError(f"folding the retelling of room {chat_id}: {folded}")
    return {"activation": activation.get("kind"), "legacy_rows": oldest, "sealed_pages": len(records),
            "folded_sections": len(fresh), "consumed_outputs": _consume_closed_outputs(root, k),
            "rotated_logs": _rotated_logs(root)}


def steady_state(root: pathlib.Path) -> Dict[str, Any]:
    """A long-lived install between the owner's turns: the maintenance child has run the jobs it owed since the
    chronicle became active (``memory_tail`` stores the facts of the rows after the frontier, so a capture reads
    them and only the rows appended since, not every row live), and the boot has observed the OpenRouter
    catalogue (``boot_catalogue``) and listed ``archive/`` for the logs a first owner turn reads
    (``boot_listings``). An install that does not reach it fails the driver: no law reads a fill."""
    from ouroboros import maintenance
    from ouroboros.chronicle_store import ChronicleStore

    jobs: Dict[str, Any] = {}
    for _attempt in range(3):
        if not maintenance.due(root):
            break
        jobs.update(maintenance.run(root, should_stop=lambda: False)["jobs"])
    tail, due = ChronicleStore(root).tail_state() or {}, maintenance.due(root)
    if not tail.get("complete") or due:
        raise RuntimeError(f"the install did not reach its steady state: tail {tail}, still due {due}, ran {jobs}")
    return {"child_jobs": jobs, "catalogue": boot_catalogue(), "listings": boot_listings(root)}


def boot_listings(root: pathlib.Path) -> Dict[str, Any]:
    """The boot's archive listing, through production (``server_boot.remember_log_listings``), its thread
    finished, as it has long before an owner's first message: the turns read the tools and events archives
    from the listing it remembers (``jsonl_tail.remembered_listing``)."""
    from ouroboros import jsonl_tail, server_boot

    server_boot.remember_log_listings(root).join()
    remembered = sorted(prefix for _dev, _ino, prefix in jsonl_tail._LISTINGS)
    if not {"events", "tools"} <= set(remembered):
        raise RuntimeError(f"the boot's archive listing did not land: {remembered}")
    return {"remembered": remembered}


def boot_catalogue() -> Dict[str, Any]:
    """The boot's catalogue observation on an install with network, through production
    (``LLMClient.prefetch_openrouter_catalog``): one clean ``/models`` answer, canned for the OpenRouter models
    this install's settings name, fills the process's catalogue and its response limits and is recorded for every
    process (``state/openrouter_models.json``)."""
    import requests

    from ouroboros.config import runtime_setting
    from ouroboros.llm import LLMClient
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS
    from ouroboros.settings_defaults import OPENROUTER_DEFAULTS

    named = [runtime_setting(key, "") for key in MODEL_ROLE_SETTINGS.values()]
    models = sorted({model for model in (*OPENROUTER_DEFAULTS.values(), *named)
                     if isinstance(model, str) and "/" in model})
    answer = {"data": [{"id": model, "context_length": CATALOGUE_WINDOW,
                        "top_provider": {"max_completion_tokens": 32_000},
                        "supported_parameters": ["max_tokens", "reasoning", "temperature", "tool_choice", "tools"]}
                       for model in models]}

    class Answer:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return answer

    real, requests.get = requests.get, lambda _url, **_kwargs: Answer()
    try:
        LLMClient.prefetch_openrouter_catalog()
    finally:
        requests.get = real
    if not LLMClient._CAPABILITIES_FETCH_OK:
        raise RuntimeError("the boot's catalogue observation did not land")
    return {"models": models}


def foreign_activity(root: pathlib.Path) -> Dict[str, Any]:
    """What a busy install does between two owner turns: a row lands in another room, a page is published
    there (the driver adds a task that completes)."""
    from ouroboros.chronicle_store import ChronicleStore
    from supervisor import message_bus

    message_bus.log_chat("out", 515151, 0, "a new row of another room", drive_root=root, require_write=True)
    ids = append_rows(root, _rows(4999, 2, "published elsewhere"))
    if not ChronicleStore(root).publish([_page(4999, ids, "a page another task published")]).ok:
        raise RuntimeError("the page published elsewhere did not land")
    return {"rooms": [515151, 4999]}


def grow_breadth(root: pathlib.Path, n: int) -> Dict[str, Any]:
    """n more live rooms with story (open rows, a page, a note) and n // 4 running roots: current breadth."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    open_rows, paged = [], {}
    for i in range(n):
        open_rows += _rows(3000 + i, OPEN_ROWS, "open")
        paged[3000 + i] = _rows(3000 + i, 2, "paged")
    append_rows(root, open_rows)
    ids = append_rows(root, [row for rows in paged.values() for row in rows])
    records = []
    for index, chat_id in enumerate(paged):
        records.append(_page(chat_id, ids[index * 2:index * 2 + 2], f"page {chat_id}"))
        records.append({"kind": "note", "room_id": str(chat_id), "task_id": f"w{chat_id}", "text": f"note of {chat_id}",
                        "author": _author(f"w{chat_id}")})
    if not ChronicleStore(root).publish(records).ok:
        raise RuntimeError("the breadth rooms' story did not land")
    for i in range(n // 4):
        task_id, text = f"cur-broad-{i:03d}", ("a running root at work with a long brief " * 1500)[:60_000]
        write_task_result(root, task_id, "running", chat_id=1, text=text, root_task_id=task_id)
        with queue._queue_lock:
            queue.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 1, "text": text[:200],
                                               "root_task_id": task_id}, "worker_id": 100 + i, "attempt": 1,
                                      "started_at": time.time()}
    return {"rooms": n, "roots": n // 4, "rows": len(open_rows) + len(ids)}


def grow_rows(root: pathlib.Path, n: int) -> Dict[str, Any]:
    """n more open rows over the rooms already live; Main's newest of them is what its next turn must carry."""
    rooms = [1] + [3000 + i for i in range(BREADTH)]
    rows = [row for i, chat_id in enumerate(rooms)
            for row in _rows(chat_id, n // len(rooms) + (i < n % len(rooms)), "more")]
    append_rows(root, rows)
    main = [row["text"] for row in rows if row["chat_id"] == 1]
    return {"rows": len(rows), "rooms": len(rooms), "main_newest": main[-1]}
