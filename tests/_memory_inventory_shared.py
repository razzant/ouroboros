"""A small installation for the memory-inventory tests: three chat rooms besides Main.

Two Projects (alpha, beta), a transport chat (777) and the hidden partition, a task
bound to alpha by an owner message written in Main, and twenty stream rows across two
archives and the live file. The legacy memory covers the first ten stream rows in two
blocks; the frontier's last covered row is the last row of the second archive.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.utils import jsonl_generation_signature

ARCHIVE_ONE = "chat_20260901T120000.jsonl"
ARCHIVE_TWO = "chat_20260902T120000.jsonl"
MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
ORIGIN = "Start alpha with the inventory work"


def msg(ts: str, text: str, *, chat_id: int = 1, direction: str = "in", **extra: Any) -> Dict[str, Any]:
    return {"chat_id": chat_id, "direction": direction, "ts": ts, "text": text, **extra}


def append(path: pathlib.Path, *rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) + "\n")


def projects(root: pathlib.Path) -> Dict[str, int]:
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    alpha = create_project(root, "alpha", name="Alpha")
    beta = create_project(root, "beta", name="Beta")
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin-a", ts="2026-09-01T00:05:00+00:00",
                                  text=ORIGIN)
    bind_task_to_project(root, "bound", alpha["id"], origin={"ref": ref, "text": ORIGIN})
    return {"alpha": int(alpha["chat_id"]), "beta": int(beta["chat_id"])}


def chat(root: pathlib.Path, rooms: Dict[str, int]) -> List[Dict[str, Any]]:
    """Twenty stream rows (positions 0-19); an A2A row, a blank and a torn line take no position."""
    a, b = rooms["alpha"], rooms["beta"]
    first = [
        msg("2026-09-01T00:00:00+00:00", "hello", client_message_id="m0"),
        msg("2026-09-01T00:01:00+00:00", "hello back", direction="out", task_id="t1"),
        msg("2026-09-01T00:02:00+00:00", "alpha question", chat_id=a, client_message_id="a0"),
        msg("2026-09-01T00:03:00+00:00", "alpha answer", chat_id=a, direction="out", task_id="ta"),
        msg("2026-09-01T00:04:00+00:00", "from the transport", chat_id=777,
            transport={"provider": "telegram", "actor": {"display_name": "Ann"}}),
        msg("2026-09-01T00:05:00+00:00", ORIGIN, client_message_id="origin-a"),
    ]
    second = [
        msg("2026-09-02T00:00:00+00:00", "bound work", direction="out", task_id="bound"),
        msg("2026-09-02T00:01:00+00:00", "child of bound", direction="out", task_id="kid", parent_task_id="bound",
            root_task_id="bound", subagent_task_id="kid", delegation_role="subagent"),
        msg("2026-09-02T00:02:00+00:00", "beta question", chat_id=b, client_message_id="b0"),
        msg("2026-09-02T00:03:00+00:00", "beta answer", chat_id=b, direction="out", task_id="tb"),
    ]
    live = [
        msg("2026-09-03T00:00:00+00:00", "Room started", direction="system", type="project_started"),
        msg("2026-09-03T00:01:00+00:00", "next please", client_message_id="m1"),
        msg("2026-09-03T00:02:00+00:00", "a long reply\n## with a heading", direction="out", task_id="t2",
            initiator="consciousness"),
        msg("2026-09-03T00:03:00+00:00", "alpha again", chat_id=a, client_message_id="a1"),
        msg("2026-09-03T00:04:00+00:00", "beta again", chat_id=b, client_message_id="b1"),
        msg("2026-09-03T00:05:00+00:00", "to the transport", chat_id=777, direction="out", task_id="t3"),
        msg("2026-09-03T00:06:00+00:00", "", direction="system", type="task_summary", task_id="t2",
            status="completed", result_ref={"reader": "get_task_result", "task_id": "t2"}),
        msg("2026-09-03T00:07:00+00:00", "hidden note", chat_id=0, direction="out"),
        msg("2026-09-03T00:08:00+00:00", "Could not start", direction="system", type="task_not_started",
            task_id="bound"),
        msg("2026-09-03T00:09:00+00:00", "and more", client_message_id="m2"),
    ]
    a2a = msg("2026-09-01T00:00:30+00:00", "agent traffic", chat_id=-5)
    append(root / "archive" / ARCHIVE_ONE, first[0], a2a, "\n", *first[1:])
    append(root / "archive" / ARCHIVE_TWO, second[0], "{torn\n", *second[1:])
    append(root / "logs" / "chat.jsonl", *live)
    return [*first, *second, *live]


def blocks(rooms: Dict[str, int]) -> List[Dict[str, Any]]:
    a, b = str(rooms["alpha"]), str(rooms["beta"])
    return [
        {"ts": "2026-09-02T01:00:00+00:00", "type": "summary", "range": "2026-09-01 00:00 - 00:05",
         "message_count": 6, "content": "### Block zero",
         "rooms": [{"room_id": "1", "label": "Main", "message_count": 4, "content": "Main talk."},
                   {"room_id": a, "label": "Alpha", "message_count": 3, "content": "Alpha began."},
                   {"room_id": "777", "label": "Transport", "message_count": 1, "content": "A transport line."}]},
        {"ts": "2026-09-02T02:00:00+00:00", "type": "summary", "range": "2026-09-02 00:00 - 00:03",
         "message_count": 4, "content": "### Block one",
         "rooms": [{"room_id": a, "label": "Alpha", "message_count": 2, "content": "Alpha worked."},
                   {"room_id": b, "label": "Beta", "message_count": 2, "content": "Beta asked."},
                   {"room_id": "1", "label": "Main", "message_count": 0, "content": "Main was quiet."}]},
    ]


def world(root: pathlib.Path, *, legacy: bool = True, cursor: bool = True, flat: str = "",
          activate: bool = True) -> Dict[str, int]:
    """The installation, optionally with the legacy memory imported.

    With the old cursor the frontier is exact at stream position 10 (the last row of the
    second archive is the last covered one); without it the frontier is unknown at the
    chain end observed at activation. ``flat`` adds a legacy ``dialogue_summary.md``.
    """
    rooms = projects(root)
    chat(root, rooms)
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    if legacy:
        (memory / "dialogue_blocks.json").write_text(json.dumps(blocks(rooms)), encoding="utf-8")
    if legacy and cursor:
        meta = {"chat_log_signature": jsonl_generation_signature(root / "archive" / ARCHIVE_TWO),
                "last_consolidated_offset": 4}
        (memory / "dialogue_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if flat:
        (memory / "dialogue_summary.md").write_text(flat, encoding="utf-8")
    if activate:
        assert ChronicleStore(root).ensure_activated()["kind"] == "activation"
    return rooms


def cold(root: pathlib.Path, min_pos: int):
    """The reference: ``chat_chain.iter_rows`` from the chain's start, at and after ``min_pos``."""
    from ouroboros import chat_chain
    from ouroboros.memory_inventory import row_meta

    return [(address, row_meta(row), pos) for address, row, pos in chat_chain.iter_rows(root) if pos >= min_pos]
